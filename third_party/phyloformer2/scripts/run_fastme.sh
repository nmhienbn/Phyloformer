#!/bin/sh

DIR=$1

if [ -z "${DIR}" ]; then
  echo "Usage:"
  echo "$0  DIR "
  echo ""
  echo "  DIR:      Directory where distance matrices are stored and where to write inferred trees"
  echo "            matrices must be in DIR/matrices"
  echo "            trees will be written in DIR/trees"
  exit 1
fi

# Make output directory
mkdir -p "${DIR}/trees"

# Infer trees
for matfile in "${DIR}/matrices/"*.phy; do
  filename="${matfile##*/}"
  fastme \
    --input_data "$matfile" \
    --output_tree "${DIR}/trees/${filename/phy/nwk}" \
    --nni --spr
done


