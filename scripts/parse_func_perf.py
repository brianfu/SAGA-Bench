# Example: python3 parse_func_perf.py 'self_perf' '6 [color="#0c8691", fontcolor="#ffffff", fontsize="10.00", label="assignLogicalID\n22.97%\n(22.97%)\n1836241×"];'
# This script should only ever perform string parsing!

# Args: {"self_perf" / "total_perf"}, rawFuncStr 
# rawFuncStr format: '6 [color="#0c8691", fontcolor="#ffffff", fontsize="10.00", label="assignLogicalID\n22.97%\n(22.97%)\n1836241×"];'
# Output: float (either self_perf or total_perf of func)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('perfType', metavar='perfType', choices=['total_perf', 'self_perf', 'func_called'],
  help='Performance: runtime incl. children, only self runtime, times function called. Allowed values: \'total_perf\', \'self_perf\', \'func_called\'')
parser.add_argument('rawFuncStr', help='Raw function string from \'gprof frontEnd | gprof2dot -s -w | grep $function\'')
args = parser.parse_args()
# args = parser.parse_args(['self_perf', '6 [color="#0c8691", fontcolor="#ffffff", fontsize="10.00", label="assignLogicalID\n22.97%\n(22.97%)\n1836241×"];'])
# args = parser.parse_args(['total_perf', '5 [color="#0c9293", fontcolor="#ffffff", fontsize="10.00", label="readBatchFromCSV\n24.83%\n(0.93%)\n19×"];'])

def main():

  # Find 'label="..."];'
  label_str_ind = args.rawFuncStr.find(f'label="')
  if (not label_str_ind):
    return(f'Error: Invalid rawFuncStr')
  
  # Get ['label="assignLogicalID', '22.97%', '(22.97%)', '1836241×"];']
  label_str = args.rawFuncStr[label_str_ind:]
  label_str_arr = label_str.split('\\n') # Bash will pass in "$rawFuncStr"; \n -> \\n

  total_perf = label_str_arr[1][:-1] # '1.1%'
  self_perf = label_str_arr[2][1:-2] # '(1.1%)'
  func_called = label_str_arr[3][:-4] # '1x"];'

  match args.perfType:
    case "total_perf":
      return total_perf
    case "self_perf":
      return self_perf
    case "func_called":
      return func_called


print(main())