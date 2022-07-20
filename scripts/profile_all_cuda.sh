# Assuming frontEnd has already been make, and Makefile contains profiling flag (-pg) and all production optimizations (e.g. -O2)
# Place script into project root (dir containing frontEnd exe)
frontEnd_dir=$PWD/..; # For run_command()

### 
# Change these to adjust ran algorithms / datastructs

# -s
datastructs=(\
  "adList" \
);

# -a
algorithms=(\
  "prfromscratch" \
  "prdyn" \
  "mcfromscratch" \
  "mcdyn" \
  "bfsfromscratch" \
  "bfsdyn" \
);

###


# -w; If given algorithm is edge weight sensitive
declare -A weights_used_alg;
weights_used_alg["prfromscratch"]=0;
weights_used_alg["prdyn"]=0;
weights_used_alg["mcfromscratch"]=0;
weights_used_alg["mcdyn"]=0;
weights_used_alg["bfsfromscratch"]=1;
weights_used_alg["bfsdyn"]=0;

# -d; If given algorithm is edge direction sensitive
# Currently, algorithms will not run if directed edges on CUDA
declare -A direction_used_ds;
direction_used_ds["prfromscratch"]=0;
direction_used_ds["prdyn"]=0;
direction_used_ds["mcfromscratch"]=0;
direction_used_ds["mcdyn"]=0;
direction_used_ds["bfsfromscratch"]=0;
direction_used_ds["bfsdyn"]=0;

# -f
declare -A datasets;
datasets["AmazonSNAP"]="$frontEnd_dir/dataset/com-amazon.ungraph.shuffle.t.w.csv";

# -b
declare -A batch_size;
batch_size["AmazonSNAP"]=50000;

# -n; Max nodes to init the datastruct with
declare -A node_size;
node_size["AmazonSNAP"]=334863;

main () {
  # Folder hirearchy should be output/{dataset}/{datastruct}/{algorithm}-{undirected/directed}-{unweighted/weighted}-{batch_size}-{node_size}
  # e.g. output/AmazonSNAP/adList/prfromscratch-undirected-unweighted-50000-334863
  # Overwrite on every new run
  cd $frontEnd_dir;
  mkdir -p output;
  cd output;
  output_root=$PWD;
  mkdir -p gprof2dot;
  gprof2dot_root="$output_root/gprof2dot"; # Rename & copy over *.png graphs here!

  for dataset_key in "${!datasets[@]}"; do
    dataset=${datasets[$dataset_key]};
    mkdir -p $dataset_key;
    cd $dataset_key;
    echo "Running $dataset_key dataset: $dataset";

    for datastruct in ${datastructs[@]}; do
      mkdir -p $datastruct
      cd $datastruct;
      echo "Running $datastruct data struct";

      for alg in ${algorithms[@]}; do
        ds_dir=$PWD; # Base folder for content subfolders
        echo "Running $alg algorithm";

        # Check if alg is weight sensitive
        if (( ${weights_used_alg[$alg]} == 1 )); then
          weight=1;
          echo "$alg algorithm is weight sensitive";
          
          directed=0;
          run_command $directed $weight;

          if (( ${direction_used_ds[$alg]} == 1 )); then
            echo "$datastruct datastruct is direction sensitive";
            directed=1;
            run_command $directed $weight;
          fi;
        fi;

        # Always run weight insensitive case
        weight=0;

        directed=0;
        run_command $directed $weight;
        
        if (( ${direction_used_ds[$alg]} == 1 )); then
          echo "$datastruct datastruct is direction sensitive";
          directed=1;
          run_command $directed $weight;
        fi;

        echo "Done running $alg algorithm";
      done;

      echo "Done running $datastruct data struct";
      cd ..;
    done;

    echo "Done running $dataset_key dataset";
    cd ..;
  done;
}

run_command () {
  local curr_directed=$1;
  local curr_weight=$2;
  local curr_directed_str="undirected";
  local curr_weight_str="unweighted";

  if (( $curr_directed == 1 )); then
    curr_directed_str="directed";
  fi;
  if (( $curr_weight == 1 )); then
    curr_weight_str="weighted";
  fi;

  # Assuming global args
  newdir="$alg-$curr_directed_str-$curr_weight_str-${batch_size[$dataset_key]}-${node_size[$dataset_key]}";
  mkdir -p $newdir;
  cd $newdir;
  run_command_inner $curr_directed $curr_weight $dataset ${batch_size[$dataset_key]} $datastruct ${node_size[$dataset_key]} $alg;
  
  png_name="$dataset_key-$datastruct-$newdir";
  cp profile.png $gprof2dot_root/$png_name.png;

  cd $ds_dir;
}

run_command_inner() {
  # Args; Same order as passed to frontEnd
  # Assuming all args are always passed!
  local directed=$1; # -d
  local weighted=$2; # -w
  local dataset=$3; # -f
  local batch_size=$4; # -b
  local datastruct=$5; # -s
  local node_size=$6; # -n
  local algorithm=$7; # -a

  # Assuming runfile is always called "frontEnd"
  $frontEnd_dir/frontEnd -d $directed -w $weighted -f $dataset -b $batch_size -s $datastruct -n $node_size -a $algorithm &> output.out
  gprof $frontEnd_dir/frontEnd > profile.out
  gprof $frontEnd_dir/frontEnd | gprof2dot -s -w | dot -Tpng -o profile.png
}

main;

# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prfromscratch &> output.out
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prfromscratch &> output.out
# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prdyn &> output.out
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prdyn &> output.out
# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a mcfromscratch &> output.out 
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a mcfromscratch &> output.out
# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a mcdyn &> output.out
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a mcdyn &> output.out
# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsfromscratch &> output.out
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsfromscratch &> output.out
# ./frontEnd -d 0 -w 1 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsfromscratch &> output.out
# ./frontEnd -d 1 -w 1 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsfromscratch &> output.out
# ./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsdyn &> output.out
# ./frontEnd -d 1 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsdyn &> output.out