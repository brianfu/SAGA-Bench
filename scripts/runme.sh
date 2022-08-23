cd ..
#dataDir=/home/abanti/datasets/SAGAdatasets/

# Run LiveJournal 
#./frontEnd -d 1 -w 0 -f ${dataDir}soc-LiveJournal1.shuffle.t.w.csv -b 500000 -s adListShared -n 4847571 -a bfsdyn -t 64 

# Run Orkut 
#./frontEnd -d 0 -w 0 -f ${dataDir}com-orkut.ungraph.shuffle.t.w.csv -b 500000 -s stinger -n 3072441 -a bfsdyn -t 64

# Run Wiki-Topcats 
#./frontEnd -d 1 -w 0 -f ${dataDir}wiki-topcats.shuffle.t.w.csv -b 500000 -s degAwareRHH -n 1791489 -a bfsdyn -t 64

# Run rmat 
#./frontEnd -d 1 -w 0 -f ${dataDir}rmat.csv -b 500000 -s adListChunked -n 33554432 -a bfsdyn -t 64

# Run Wiki-Talk
#./frontEnd -d 1 -w 0 -f ${dataDir}wiki-talk-pure.shuffle.t.w.csv -b 500000 -s adListChunked -n 2394385 -a bfsdyn -t 64

#Run test.csv 
#./frontEnd -d 1 -w 1 -f ./test.csv -b 10 -t 24 -s stinger -n 40 -a prdyn

#Run test.csv on algorithm "traverse". "Traverse" is a single-threaded micro-kernel that reads the neighbors of all vertices in the graph
#./frontEnd -d 1 -w 1 -f ./test.csv -b 10 -t 24 -s stinger -n 40 -a traverse

# Amazon
./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adListShared -n 334863 -a mcdyn &> output.out
gprof frontEnd > profile.out
gprof frontEnd | gprof2dot -s -w | dot -Tpng -o profile.png

#./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a bfsfromscratch &> output.out
#./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prdyn &> output.out
#./frontEnd -d 0 -w 0 -f ./dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a prfromscratch &> output.out

### gem5
# cd ../gem5;
# > nohup.out;
# scons build/X86/gem5.opt -j97

# nohup build/X86/gem5.opt configs/example/se.py --cmd=../SAGA-Bench/frontEnd \
# -o '-d 0 -w 0 -f ../SAGA-Bench/dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adList -n 334863 -a mcdyn' \
# --cpu-type=TimingSimpleCPU --l1d_size=64kB --l1i_size=16kB --num-cpus 4 --caches --l2cache --output=output.out --errout=error.out;

# nohup build/X86/gem5.opt configs/example/se.py --cmd=../SAGA-Bench/frontEnd \
# -o '-d 0 -w 0 -f ../SAGA-Bench/dataset/com-amazon.ungraph.shuffle.t.w.csv -b 50000 -s adListShared -n 334863 -a bfsdyn' \
# --fast-forward=1420000 --maxinsts=180000000 \
# --cpu-type=TimingSimpleCPU --l1d_size=64kB --l1i_size=16kB --num-cpus 4 --caches --l2cache --output=output.out --errout=error.out;

#Tried algs: bfsdyn, bfsfromscratch,
#To try alg: prdyn, prfromscratch, ccdyn, ccfromscratch, mcdyn, mcfromscratch, ssspdyn, ssspfromscratch, sswpdyn, sswpfromscratch
#Tried DS: 
#DS: adListShared, adListChunked, stinger, degAwareRHH