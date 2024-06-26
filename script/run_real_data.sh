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
# # The following code is for the real data
sequence_exp_id=(1)
mkdir ./result/real_data_1

for id in "${sequence_exp_id[@]}"; do
	for i in {1..100}; do
		terminal_width=$(tput cols)
		while true; do
			current_running_processes=$(ps -eo s= | grep -c 'R')
			free_memory_percent=$(get_free_memory_percent)
			used_memory_percent=$((100 - free_memory_percent))
			printf "\e[1;34m%*s\e[0m\r" $terminal_width "[loop=$i; ${used_memory_percent}%Mem]"
			if [ $current_running_processes -lt $max_running_processes ] && [ $free_memory_percent -ge 20 ]; then
				nohup python real_data.py --seed "$i" --exp_id "$id" --record_dir "./result/real_data_1" --verbose 0 >> "./result/real_data_1/out.txt" 2>&1 &
				sleep_time=$((1))
				sleep $sleep_time
				break
				# Exit the while loop after starting the Python process
			else
				sleep 5
			fi
		done
		rm ./result/real_data_1/out.txt
	done
done


mkdir ./result/real_data_3
sequence_exp_id=(3)

for id in "${sequence_exp_id[@]}"; do
	for i in {1..100}; do
		terminal_width=$(tput cols)
		while true; do
			current_running_processes=$(ps -eo s= | grep -c 'R')
			free_memory_percent=$(get_free_memory_percent)
			used_memory_percent=$((100 - free_memory_percent))
			printf "\e[1;34m%*s\e[0m\r" $terminal_width "[loop=$i; ${used_memory_percent}%Mem]"
			if [ $current_running_processes -lt $max_running_processes ] && [ $free_memory_percent -ge 20 ]; then
				nohup python real_data.py --seed "$i" --exp_id "$id" --record_dir "./result/real_data_3" --verbose 0 >> "./result/real_data_3/out.txt" 2>&1 &
				sleep_time=$((1))
				sleep $sleep_time
				break
				# Exit the while loop after starting the Python process
			else
				sleep 5
			fi
		done
		rm ./result/real_data_3/out.txt
	done
done
