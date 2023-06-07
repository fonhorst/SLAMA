package org.apache.spark.lightautoml.utils

import org.apache.spark
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import scala.collection.JavaConverters._
import scala.util.Random

class TestFullCoalescer extends AnyFunSuite with BeforeAndAfterEach with Logging {
  val folds_count = 5
  val workers_nums = List(1, 2, 3, 4, 5)
  val cores_nums = List(1, 2, 3, 4, 5)
  val wc_nums: List[(Int, Int)] = workers_nums.flatMap(x => cores_nums.map(y => (x, y)))

  override protected def afterEach(): Unit = {
    SparkSession.getActiveSession.foreach(spark=> spark.stop())
  }

  wc_nums.foreach {
    case (num_workers, num_cores) =>
      test(s"Coalescers for num_workers=$num_workers and num_cores=$num_cores") {
        val spark = SparkSession
                .builder()
                .master(s"local-cluster[$num_workers, $num_cores, 1024]")
                //          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar,target/scala-2.12/spark-lightautoml_2.12-0.1.1-tests.jar")
                .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar")
                .config("spark.default.parallelism", "6")
                .config("spark.sql.shuffle.partitions", "6")
                .config("spark.locality.wait", "15s")
                .getOrCreate()

        import spark.sqlContext.implicits._

        val df = spark
                .sparkContext.parallelize((0 until 100)
                .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
                .repartition(num_workers * num_cores * 2)
                .cache()
        df.write.mode("overwrite").format("noop").save()

        val all_elements = df.select("data").collect().map(row => row.getAs[Int]("data")).toList

        val (dfs, base_rdd) = SomeFunctions.test_full_coalescer(df, 3, materialize_base_rdd = true)

        dfs.asScala.foreach(df =>{
          val df_elements = df.select("data").collect().map(row => row.getAs[Int]("data")).toList
          df_elements should contain theSameElementsAs all_elements
        })

        base_rdd.unpersist()

        spark.stop()
      }
  }
}
