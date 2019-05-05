import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StandardScaler, VectorAssembler}

// Solo mostrar log de errores en la consola
val rootLogger = Logger.getRootLogger()
rootLogger.setLevel(Level.ERROR)

// Lectura de datos (fichero descargado de kaggle, el fichero de blackboard no se descargaba).
val dirty_df = spark.read.format("csv").option("header", "true").option("delimiter", ",").option("inferSchema", "true").load("weatherAUS.csv")

/*
	No se infiere el esquema correctamente, así que se procede a hacer las transformaciones de manera manual.
	
	1. Ya que existen más columnas no-numéricas, se crea una lista con sus etiquetas.
	2. Se hace un diff entre todas las columnas del Dataframe y las columnas no-numéricas para obtener
	   aquellas que deben ser transformadas a Double.
	3. Se realiza un foldLeft para recorrer la lista de columnas que deben ser de tipo Double y estas
	   se castean a DoubleType
	4. Para imputar los valores nulos no ha funcionado la misma idea del foldLeft así que se ha hecho
	   de manera manual siguiendo las instrucciones dadas para cada tipo de columna.
*/
val not_double_cols = Array("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow")
val double_cols = (dirty_df.schema.fieldNames.toArray diff not_double_cols).map(c => c.toString).toArray

val weather_nulls_df = double_cols.foldLeft(
	dirty_df
) {(aux_df, col_name) => aux_df.withColumn(
		col_name, aux_df.col(col_name).cast(DoubleType)
	)
}

val weather_df = weather_nulls_df
.withColumn("MinTemp", coalesce(col("MinTemp"), lit(weather_nulls_df.agg(mean("MinTemp").alias("MinTemp")).select("MinTemp").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("MaxTemp", coalesce(col("MaxTemp"), lit(weather_nulls_df.agg(mean("MaxTemp").alias("MaxTemp")).select("MaxTemp").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Rainfall", coalesce(col("Rainfall"), lit(weather_nulls_df.agg(mean("Rainfall").alias("Rainfall")).select("Rainfall").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Evaporation", coalesce(col("Evaporation"), lit(weather_nulls_df.agg(mean("Evaporation").alias("Evaporation")).select("Evaporation").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Sunshine", coalesce(col("Sunshine"), lit(weather_nulls_df.agg(mean("Sunshine").alias("Sunshine")).select("Sunshine").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("WindGustSpeed", coalesce(col("WindGustSpeed"), lit(weather_nulls_df.agg(mean("WindGustSpeed").alias("WindGustSpeed")).select("WindGustSpeed").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("WindSpeed9am", coalesce(col("WindSpeed9am"), lit(weather_nulls_df.agg(mean("WindSpeed9am").alias("WindSpeed9am")).select("WindSpeed9am").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("WindSpeed3pm", coalesce(col("WindSpeed3pm"), lit(weather_nulls_df.agg(mean("WindSpeed3pm").alias("WindSpeed3pm")).select("WindSpeed3pm").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Humidity9am", coalesce(col("Humidity9am"), lit(weather_nulls_df.agg(mean("Humidity9am").alias("Humidity9am")).select("Humidity9am").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Humidity3pm", coalesce(col("Humidity3pm"), lit(weather_nulls_df.agg(mean("Humidity3pm").alias("Humidity3pm")).select("Humidity3pm").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Pressure9am", coalesce(col("Pressure9am"), lit(weather_nulls_df.agg(mean("Pressure9am").alias("Pressure9am")).select("Pressure9am").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Pressure3pm", coalesce(col("Pressure3pm"), lit(weather_nulls_df.agg(mean("Pressure3pm").alias("Pressure3pm")).select("Pressure3pm").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Cloud9am", coalesce(col("Cloud9am"), lit(weather_nulls_df.agg(mean("Cloud9am").alias("Cloud9am")).select("Cloud9am").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Cloud3pm", coalesce(col("Cloud3pm"), lit(weather_nulls_df.agg(mean("Cloud3pm").alias("Cloud3pm")).select("Cloud3pm").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Temp9am", coalesce(col("Temp9am"), lit(weather_nulls_df.agg(mean("Temp9am").alias("Temp9am")).select("Temp9am").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("Temp3pm", coalesce(col("Temp3pm"), lit(weather_nulls_df.agg(mean("Temp3pm").alias("Temp3pm")).select("Temp3pm").take(1)(0).get(0).asInstanceOf[Double])))
.withColumn("WindGustDir", coalesce(col("WindGustDir"), lit("unknown")))
.withColumn("WindDir9am", coalesce(col("WindDir9am"), lit("unknown")))
.withColumn("WindDir3pm", coalesce(col("WindDir3pm"), lit("unknown")))
.withColumn("RainToday", coalesce(col("RainToday"), lit("unknown")))
.withColumn("label", coalesce(col("RainTomorrow"), lit("unknown")))

// Se verifica que en la salida no existan valores nulos
weather_df.filter(col("label") === "unknown").agg(count("*")).show

// Función para la creación y ejecución de todos los modelos
def models(categorical_feats: Array[String], numeric_feats: Array[String], train_df: Dataset[Row], test_df: Dataset[Row]): PipelineModel = {
	// Se crea un indexador para las variables categóricas
	val indexers = (categorical_feats ++ Array("label")).map { col_name =>
		new StringIndexer().setInputCol(col_name).setOutputCol(col_name + "Index")
	}

	val categories_indexed = categorical_feats.map(col_name => col_name + "Index")
	// Se unen todas las features en un vector para cada fila del dataset
	val assembler = new VectorAssembler().setInputCols(categories_indexed ++ numeric_feats).setOutputCol("features")

	// Se normalizan los valores
	val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").setWithStd(true).setWithMean(true)

	// Se crea el modelo de clasificación
	val lr = new LogisticRegression().setMaxIter(10).setLabelCol("labelIndex")

	// Se añaden todos los modelos a un pipeline
	val pipeline = new Pipeline().setStages(indexers ++ Array(assembler, scaler, lr))

	// Entrenamiento
	val model = pipeline.fit(train_df)
	// Test
	val predictions = model.transform(test_df)
	val predictions_train = model.transform(train_df)
	val evaluator = new BinaryClassificationEvaluator().setLabelCol("labelIndex").setRawPredictionCol("rawPrediction")
    val train_auc = evaluator.evaluate(predictions_train)
	val test_auc = evaluator.evaluate(predictions)

    println("Model performance:\n - AUC(train) = " + train_auc + "\n - AUC(test) = " + test_auc + "\n")

	model
}

// Función para crear un modelo y hacer una evaluación para una ciudad especificada
def evaluate_by_city(data: Dataset[Row], categorical_feats: Array[String], numeric_feats: Array[String], city: String): PipelineModel = {
	println("\n Creating model for: %s\n".format(city))
	val Array(train_df, test_df) = data.filter(col("Location") === city).randomSplit(Array(0.8, 0.2))
	models(categorical_feats, numeric_feats, train_df, test_df)
}

// Función para predecir si mañana lloverá en alguna ciudad de Australia
def rain_tomorrow(today_data: Dataset[Row], model: PipelineModel): Double = {
	// Se obtiene la predicción para el día de mañana
	val predictions = model.transform(today_data)

	// Se crea una función para transformar un Vector a un Array
	val vecToArray = udf((xs: Vector) => xs.toArray)

	// Se crea una columna para cada elemento del vector y luego se consulta la nombrada 'Yes'
	// que contiene la probabilidad de que sí llueva mañana
	val new_df = predictions.withColumn("probArr" , vecToArray(col("probability")))
	val elements = Array("No", "Yes")
	val sqlExpr = elements.zipWithIndex.map{case (alias, idx) => col("probArr").getItem(idx).as(alias)}
	val pred = new_df.select(sqlExpr : _*)
	pred.select("Yes").take(1)(0).get(0).asInstanceOf[Double]
}

val categorical_feats = Array("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")
val canberra_model = evaluate_by_city(weather_df, categorical_feats, double_cols, "Canberra")
val any_city_model = evaluate_by_city(weather_df, categorical_feats, double_cols, "Albury")
println("\nRain tomorrow in Canberra?... %3.2f%%\n".format(rain_tomorrow(weather_df.filter(col("Location") === "Canberra").limit(1), canberra_model) * 100))
println("\nRain tomorrow in Albury?... %3.2f%%\n".format(rain_tomorrow(weather_df.filter(col("Location") === "Albury").limit(1), any_city_model) * 100))
