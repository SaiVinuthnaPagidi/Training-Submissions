# toy example: analyzing F1 pit stop times
pit_stops = {
    "Red Bull": [2.1, 2.3, 2.0],
    "Ferrari": [2.7, 2.5, 2.6],
    "McLaren": [2.2, 2.1, 2.3],
    "Mercedes": [2.4, 2.3, 2.5]
}

# calculating each team's average pit stop time
average_times = {team: sum(times)/len(times) for team, times in pit_stops.items()}

# find the fastest team
fastest_team = min(average_times, key=average_times.get)

# display results
print("🏎️ F1 Pit Stop Analysis 🏎️")
for team, avg in average_times.items():
    print(f"{team}: average pit stop time = {avg:.2f} seconds")

print(f"\n🏆 Fastest Pit Crew: {fastest_team} with an average of {average_times[fastest_team]:.2f} seconds!")
