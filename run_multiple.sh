#!/bin/zsh

# Get the number of CPU cores in MacOS
max_running_processes=3

sequence=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 60 80 100)

for j in "${sequence[@]}"; do
    for i in {1..50}; do
        terminal_width=$(tput cols)
        printf "\e[1;34m%*s\e[0m\r" $terminal_width "[Processing: p=$j, loop=$i]"
        while true; do
            # Modify this line for macOS compatibility
            current_running_processes=$(ps aux | grep python | grep nonlinear | grep -v grep | wc -l)
            if [ $current_running_processes -lt $max_running_processes ]; then
                nohup python demo_nonlinear_v3.5.py --p "$j" --seed "$i" --record_dir "./result/result_v3.5.5.1_tmp" --verbose 0 >> "./result/result_v3.5.5.1_tmp/out.txt" 2>&1 &
                break # Exit the while loop after starting the Python process
            else
                sleep 5
            fi
        done
    done
done
