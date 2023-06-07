package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
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
              .sparkContext.parallelize((0 until 5)
              .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
              .repartition(num_workers * num_cores * 2)
              .cache()
      df.write.mode("overwrite").format("noop").save()

      val (dfs, base_rdd) = SomeFunctions.test_full_coalescer(df, 3)

      dfs.asScala.foreach(df =>{
         df.write.mode("overwrite").format("noop").save()
      })

      val k = 0
    }
}
