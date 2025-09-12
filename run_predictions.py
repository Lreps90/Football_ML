import prediciton_pipeline as pp
import time

start = time.time()

end_date = "2025-09-14"

print("\n### ---- OVER HT 0.5+ PREDICTIONS --- ###")
pp.run_2h_htscore(end_date)


print("\n### ---- UNDER 2.5 PREDICTIONS --- ###\n")
pp.run_u25(end_date)


print("\n### ---- OVER 2.5 PREDICTIONS --- ###\n")
pp.run_o25(end_date)



end = time.time()

elapsed_time = end - start  # Calculate elapsed time in seconds

# Print the elapsed time in seconds, minutes, and hours:
print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))