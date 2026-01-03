import prediciton_pipeline as pp
import time
from datetime import date, timedelta

start = time.time()

today = date.today()
today_day_of_week = today.strftime("%A")          # e.g. "Sunday"
end_date = (today + timedelta(days=1)).isoformat() # e.g. "2025-12-21"

print("Today:", today_day_of_week)
print("end_date:", end_date)


print("\n### ---- OVER HT 0.5+ PREDICTIONS --- ###")
pp.run_2h_htscore(end_date)


print("\n### ---- UNDER 2.5 PREDICTIONS --- ###\n")
pp.run_u25(end_date)


print("\n### ---- OVER 2.5 PREDICTIONS --- ###\n")
pp.run_o25(end_date)

print("\n### ---- LAY HOME IMPORT --- ###\n")
lay_home_pkl_path = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results\Lay_Home\model_file\best_model_LAY_HOME_xgb_calibrated_20251102_214705.pkl"
pp.run_lay_home(end_date, lay_home_pkl_path)

print("\n### ---- LAY AWAY IMPORT --- ###\n")
lay_away_pkl_path = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results\Lay_Away\model_file\best_model_LAY_AWAY_mlp_calibrated_20251207_175005.pkl"
pp.run_lay_away(end_date, lay_away_pkl_path)

print("\n### ---- LAY DRAW IMPORT --- ###\n")
lay_draw_pkl_path = r"C:\Users\leere\PycharmProjects\Football_ML3\FT Results\Lay_Draw\model_file\best_model_LAY_DRAW_mlp_calibrated_20251207_202353.pkl"
pp.run_lay_draw(end_date, lay_draw_pkl_path)


end = time.time()

elapsed_time = end - start  # Calculate elapsed time in seconds

# Print the elapsed time in seconds, minutes, and hours:
print("Elapsed time in seconds: {:.2f}".format(elapsed_time))
print("Elapsed time in minutes: {:.2f}".format(elapsed_time / 60))