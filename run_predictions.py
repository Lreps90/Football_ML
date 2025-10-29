import prediciton_pipeline as pp
import time

start = time.time()

end_date = "2025-10-30"

print("\n### ---- OVER HT 0.5+ PREDICTIONS --- ###")
pp.run_2h_htscore(end_date)


print("\n### ---- UNDER 2.5 PREDICTIONS --- ###\n")
pp.run_u25(end_date)


print("\n### ---- OVER 2.5 PREDICTIONS --- ###\n")
pp.run_o25(end_date)

print("\n### ---- LAY HOME IMPORT --- ###\n")
lay_home_pkl_path = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results\Lay_Home\model_file\best_model_LAY_HOME_xgb_calibrated_20251026_173444.pkl"
pp.run_lay_home(end_date, lay_home_pkl_path)

print("\n### ---- LAY AWAY IMPORT --- ###\n")
lay_away_pkl_path = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results\Lay_Away\model_file\best_model_LAY_AWAY_xgb_calibrated_20251025_215848.pkl"
pp.run_lay_away(end_date, lay_away_pkl_path)


end = time.time()

elapsed_time = end - start  # Calculate elapsed time in seconds

# Print the elapsed time in seconds, minutes, and hours:
print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))