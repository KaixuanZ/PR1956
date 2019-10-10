# Declare ad pages
declare -a AdPage=("pr1954_p0939_0.json")

Path='../../results/personnel-records/1954/seg/supplement/page_rect/'

echo "Removing Empty Page"

for json in "${AdPage[@]}"; do
  echo "Removing ${Path}$json"
  rm ${Path}$json
done

declare -a AdPage=("pr1954_p0053_0.json" )

Path='../../results/personnel-records/1954/seg/official_office/page_rect/'

echo "Removing Empty Page"

for json in "${AdPage[@]}"; do
  echo "Removing ${Path}$json"
  rm ${Path}$json
done