#!/bin/bash

Help(){
	text='''
	Usage: $(basename $0) [OPTIONS]

	Options:
	-m | --make             Make frontEnd
	-c | --copy             Copy frontEnd & runme.sh to remote machine
	-r | --run              Run runme.sh on remote machine
	-g | --get_results      Get results from run from remote machine (and place into curr dir)
	'''

	echo "$text"
}

# Current GPU server we are running tests on
# Make sure we can ssh in manually before attempting to SCP! (Requires SFU VPN)
# https://www.sfu.ca/computing/about/support/covid-19-response--working-remotely/csil-linux-remote-access.html
curr_user="fubof";
curr_server="asb9700u-i03.csil.sfu.ca";
curr_port="24";
remote_dir="~/runfiles";


Copy_runfiles()
{
	# Setup passwordless SSH and remote output dir
	ssh-copy-id -p $curr_port $curr_user@$curr_server &> /dev/null;
	ssh -p $curr_port $curr_user@$curr_server "mkdir -p $remote_dir";

	scp -P $curr_port ./frontEnd $curr_user@$curr_server:$remote_dir;
	scp -P $curr_port ./runme.sh $curr_user@$curr_server:$remote_dir;

	# Copy over all necessary input graph .csv's
	scp -P $curr_port ./test.csv $curr_user@$curr_server:$remote_dir;
}

Run_on_remote()
{
	ssh -p $curr_port $curr_user@$curr_server "$remote_dir/runme.sh";
}

Get_from_remote()
{
	scp -P $curr_port $curr_user@$curr_server:$remote_dir/Update.csv $PWD;
}


Main(){
	cd ~/code/SAGA-Bench;

	if [[ $# -eq 0 ]]; then
		Help;
	fi

	while [[ $# -gt 0 ]]; do
		case $1 in
			-m | --make)
				make;
				shift
			;;

			-c | --copy)
				Copy_runfiles;
				shift
			;;

			-r | --run)
				Run_on_remote;
				shift
			;;

			-g | --get_results)
				Get_from_remote;
				shift
			;;


			*)
				shift
			;;
		esac
	done
}

Main $@