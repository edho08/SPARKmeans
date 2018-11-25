#TIMESTAMP
timestamp() {
  date +"%F %T "
}
. ./banner.sh
. ./config.sh
echo "$(timestamp) CONFIG DONE"
echo "$(timestamp) CLEANING PREVIOUS USE"
rm -r "$OUTPUT" >/dev/null 2>&1
hdfs dfs -rm -r "$OUTPUT"  >/dev/null 2>&1
echo "$(timestamp) CLEANING DONE"
spark-submit py/KMEANS.py $INPUT $ITER $CLUSTERNUM $OUTPUT $spark_context
