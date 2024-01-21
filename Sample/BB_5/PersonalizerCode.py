def normalize(value):
    return value / 10.0  # Normalize values to the range [0, 1]

def weightsCalculation(factor):
    temp = 0
    if 0.0 <= factor <= 0.3:
        temp =  0.15
    elif 0.4 <= factor <= 0.7:
        temp = 0.55
    elif 0.8 <= factor <= 1.0:
        temp = 0.9
    return temp

def calculate_emotional_score(depression, anxiety, sleep_disturbance, substance_intake):
    normalized_depression = normalize(depression)
    normalized_anxiety = normalize(anxiety)
    normalized_sleep = normalize(sleep_disturbance)
    normalized_substance = normalize(substance_intake)
    
    weight_depression = weightsCalculation(normalized_depression)
    weight_sleep = weightsCalculation(normalized_sleep)
    weight_anxiety = weightsCalculation(normalized_anxiety)
    weight_substance = weightsCalculation(normalized_substance)

    emotional_score = (weight_depression * normalized_depression +
                       weight_sleep * normalized_sleep +
                       weight_anxiety * normalized_anxiety +
                       weight_substance * normalized_substance) / (weight_depression+ weight_sleep + weight_anxiety + weight_substance)

    return emotional_score

def calculate_weighted_output(emotional_score):
    if 0.0 <= emotional_score <= 0.3:
        return emotional_score * 0.15
    elif 0.4 <= emotional_score <= 0.7:
        return emotional_score * 0.55
    elif 0.8 <= emotional_score <= 1.0:
        return emotional_score * 0.9

# Get user input
depression_rate = float(input("Enter depression rate (1-10): "))
anxiety_rate = float(input("Enter anxiety rate (1-10): "))
sleep_disturbance_rate = float(input("Enter sleep disturbance rate (1-10): "))
substance_intake_rate = float(input("Enter substance intake rate (1-10): "))

# Calculate emotional score
emotional_score = calculate_emotional_score(depression_rate, anxiety_rate, sleep_disturbance_rate, substance_intake_rate)

# Calculate weighted output
#weighted_output = calculate_weighted_output(emotional_score)

# Print results
print(f"\nEmotional Score: {emotional_score}")
#print(f"Weighted Output: {weighted_output}")