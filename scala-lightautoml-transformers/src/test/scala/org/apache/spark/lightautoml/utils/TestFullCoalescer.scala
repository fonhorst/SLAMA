package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.sum
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import scala.collection.JavaConverters._
import scala.util.Random

class TestFullCoalescer extends AnyFunSuite with BeforeAndAfterAll with Logging {
    val num_workers = 3
    val num_cores = 2
    val folds_count = 5

    val spark: SparkSession = SparkSession
            .builder()
            .master(s"local-cluster[$num_workers, $num_cores, 1024]")
            //          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar,target/scala-2.12/spark-lightautoml_2.12-0.1.1-tests.jar")
            .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar")
            .config("spark.default.parallelism", "6")
            .config("spark.sql.shuffle.partitions", "6")
            .config("spark.locality.wait", "15s")
            .getOrCreate()

    override protected def afterAll(): Unit = {
      spark.stop()
    }

    test("Coalescers") {
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
    }
}
