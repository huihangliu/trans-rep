#!/bin/zsh
# Get the number of CPU cores and subtract 10
max_running_processes=$(( $(nproc) - 5 ))


# Function to check if there is enough free memory (5GB)
get_free_memory_percent() {
	local total_memory_mb=$(free -m | awk '/^Mem:/{print $2}')
	local used_memory_mb=$(free -m | awk '/^Mem:/{print $3}')
	local free_memory_percent=$((100 - (used_memory_mb * 100 / total_memory_mb)))

	echo $free_memory_percent # Output the free memory percentage

	if [ $free_memory_percent -ge 20 ]; then # e.g., 20% free memory required
		return 0 # True, enough memory available
	else
		return 1 # False, not enough memory
	fi
}

# --------------------------------------------------------------------------------
# # The following code is for the asymptotic variance simulation
for case in 1 2 3 4 5 6 7 8; do
	if [ $case -le 4 ]; then
		n_source=200
		p=20
	else
		n_source=400
		p=10
	fi
	if [ $case -eq 1 ] || [ $case -eq 5 ]; then
		expid=6
	fi
	if [ $case -eq 2 ] || [ $case -eq 6 ]; then
		expid=8
	fi
	if [ $case -eq 3 ] || [ $case -eq 7 ]; then
		expid=9
	fi
	if [ $case -eq 4 ] || [ $case -eq 8 ]; then
		expid=10
	fi
	for i in {1..1000}; do
		terminal_width=$(tput cols)
		while true; do
			current_running_processes=$(ps -eo s= | grep -c 'R')
			free_memory_percent=$(get_free_memory_percent)
			used_memory_percent=$((100 - free_memory_percent))
			printf "\e[1;34m%*s\e[0m\r" $terminal_width "[case=$case; loop=$i; ${used_memory_percent}%Mem]"
			if [ $current_running_processes -lt $max_running_processes ] && [ $free_memory_percent -ge 20 ]; then
				nohup python exp3_asy_normality.py --exp_id "$expid" --n_source "$n_source" --p "$p" --q 5 --r 5 --T 20 --seed "$i" --record_dir "./result/exp3_asy_normality" --verbose 0 >> "./result/exp3_asy_normality/out.txt" 2>&1 &
				sleep_time=$((1))
				sleep $sleep_time
				break
				# Exit the while loop after starting the Python process
			else
				sleep 5
			fi
		done
		rm ./result/exp3_asy_normality/out.txt
	done
done
