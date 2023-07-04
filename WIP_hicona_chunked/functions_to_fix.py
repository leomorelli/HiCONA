# FOR LONG DISTANCE DECAY
# from numpy import exp
# def decay_function(x, a, b, c):
#     return a * exp(-b * x) + c
# decay_curve = group_counts.agg(stat).to_frame()
# decay_curve.reset_index(inplace=True)
# decay_curve.plot(x="bin_difference", y="count")
# plt.show()

# popt, _ = curve_fit(
#     decay_function, decay_curve["bin_difference"], decay_curve["count"]
# )
# plt.plot(
#     decay_curve["bin_difference"],
#     decay_function(decay_curve["bin_difference"], *popt),
# )

# ALTERNATIVE WRAPPER
# def integral_wrapper(*args, **kwargs):
#     int_key = str(args) + str(kwargs)
#     int_val = int_cache.get(int_key)
#     if not int_val:
#         int_val = func(*args, **kwargs)
#         int_cache[int_key] = int_val
#     return int_val
