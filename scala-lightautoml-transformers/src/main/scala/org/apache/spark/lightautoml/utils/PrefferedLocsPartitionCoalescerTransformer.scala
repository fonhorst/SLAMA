package org.apache.spark.lightautoml.utils

import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.{CoalescedRDD, PartitionPruningRDD, RDD, ShuffledRDD}
import org.apache.spark.sql.functions.{array, col, explode, lit, rand}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.StructType

import scala.collection.JavaConverters._

class PrefferedLocsPartitionCoalescerTransformer(override val uid: String,
                                                 val prefLocs: List[String],
                                                 val do_shuffle: Boolean) extends Transformer  {

  def this(uid: String, prefLocs: java.util.List[String], do_shuffle: Boolean) = this(uid, prefLocs.asScala.toList, do_shuffle)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.active
    val ds = dataset.asInstanceOf[Dataset[Row]]
    val master = spark.sparkContext.master

    val foundCoresNum = spark.conf.getOption("spark.executor.cores") match {
      case Some(cores_option) => Some(cores_option.toInt)
      case None => if (master.startsWith("local-cluster")){
        val cores = master.slice("local-cluster[".length, master.length - 1).split(',')(1).trim.toInt
        Some(cores)
      } else if (master.startsWith("local")) {
        val num_cores = master.slice("local[".length, master.length - 1)
        val cores = if (num_cores == "*") { java.lang.Runtime.getRuntime.availableProcessors } else { num_cores.toInt }
        Some(cores)
      } else {
        None
      }
    }

    assert(foundCoresNum.nonEmpty, "Cannot find number of used cores per executor")

    val execCores = foundCoresNum.get

    val coalesced_rdd = ds.rdd.coalesce(
      numPartitions = execCores * prefLocs.size,
      shuffle = do_shuffle,
      partitionCoalescer = Some(new PrefferedLocsPartitionCoalescer(prefLocs))
    )

    spark.createDataFrame(coalesced_rdd, schema = dataset.schema)
  }

  override def copy(extra: ParamMap): Transformer = new PrefferedLocsPartitionCoalescerTransformer(uid, prefLocs, do_shuffle)

  override def transformSchema(schema: StructType): StructType = schema.copy()
}

class TrivialPartitioner(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}


object SomeFunctions {
  def func[T](x: T): T = x

  def executors(): java.util.List[java.lang.String] = {
    import scala.collection.JavaConverters._
    if (SparkSession.active.sparkContext.master.startsWith("local[")) {
      SparkSession.active.sparkContext.env.blockManager.master.getMemoryStatus
              .map { case (blockManagerId, _) => blockManagerId }
              .map { executor => s"executor_${executor.host}_${executor.executorId}" }
              .toList.asJava
    } else {
      SparkSession.active.sparkContext.env.blockManager.master.getMemoryStatus
              .map { case (blockManagerId, _) => blockManagerId }
              .filter(_.executorId != "driver")
              .map { executor => s"executor_${executor.host}_${executor.executorId}" }
              .toList.asJava
    }
  }

  def test_func(df: DataFrame): Long = {
    df.rdd.barrier().mapPartitions(SomeFunctions.func).count()
  }

  def test_full_coalescer(df: DataFrame, numSlots: Int): (java.util.List[DataFrame], RDD[Row]) = {
    // prepare and identify params for slots
    val spark = SparkSession.active
    val master = spark.sparkContext.master
    val execs = SomeFunctions.executors()
    val numExecs = execs.size()
    val foundCoresNum = spark.conf.getOption("spark.executor.cores") match {
      case Some(cores_option) => Some(cores_option.toInt)
      case None => if (master.startsWith("local-cluster")){
        val cores = master.slice("local-cluster[".length, master.length - 1).split(',')(1).trim.toInt
        Some(cores)
      } else if (master.startsWith("local")) {
        val num_cores = master.slice("local[".length, master.length - 1)
        val cores = if (num_cores == "*") { java.lang.Runtime.getRuntime.availableProcessors } else { num_cores.toInt }
        Some(cores)
      } else {
        None
      }
    }

    val numPartitions = numExecs * foundCoresNum.get

    assert(numPartitions % numSlots == 0, "Resulting num partitions should be exactly dividable by num slots")
    assert(numExecs % numSlots == 0, "Resulting num executors should be exactly dividable by num slots")

    val partitionsPerSlot = (numPartitions / numSlots)
    val numExecsPerSlot = numExecs / numSlots
    val prefLocsForSlots = (0 until numSlots)
            .map(slot_id => execs.subList(slot_id * numExecsPerSlot, (slot_id + 1) * numExecsPerSlot).asScala.toList)

    // prepare the initial dataset by duplicating its content and assigning partition_id for a specific duplicated rows
    val duplicated_df = df
            .withColumn(
              "__partition_id",
              explode(array((0 until numSlots).map(x => lit(x)):_*))
            )
            .withColumn(
              "__partition_id",
              col("__partition_id") * lit(partitionsPerSlot)
                      + (lit(partitionsPerSlot) * rand(seed = 42)).cast("int")
            )

    // repartition the duplicated dataset to force all desired copies into specific subsets of partitions
    // should work even with standard HashPartitioner
    val new_rdd = new ShuffledRDD[Int, Row, Row](
      duplicated_df.rdd.map(row => (row.getInt(row.fieldIndex("__partition_id")), row)),
      new TrivialPartitioner(numPartitions)
    ).map(x => x._2)

    // not sure if it is needed or not to perform all operation in parallel
    val copies_rdd = new_rdd.cache()
    copies_rdd.count()
    // alternative
//    val copies_rdd = new_rdd

    // select subsets of partitions that contains independent copies of the initial dataset
    // assign it preferred locations and convert the resulting rdds into DataFrames
    val prefLocsDfs = (0 until numSlots).zip(prefLocsForSlots)
            .map {
              case (slotId, prefLocs) =>
                new PartitionPruningRDD(copies_rdd, x => x / partitionsPerSlot == slotId).coalesce(
                  numPartitions = partitionsPerSlot,
                  shuffle = false,
                  partitionCoalescer = Some(new PrefferedLocsPartitionCoalescer(prefLocs))
                )
            }
            .map(rdd => spark.createDataFrame(rdd, schema = duplicated_df.schema).drop("__partition_id"))
            .toList
            .asJava

    (prefLocsDfs, copies_rdd)
  }
}