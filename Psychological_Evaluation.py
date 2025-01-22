import json

def psychological_evaluation():
    print("Welcome to the Psychological Evaluation")
    print("Please answer the following questions honestly.")
    print("Scoring is as follows: 5 = Excellent, 4 = Good, 3 = Average, 2 = Poor, 1 = Very Poor")
    
    # Questions for evaluation
    questions = [
        "On a scale of 1 to 5, how often do you feel stressed in a day?",
        "On a scale of 1 to 5, how well do you sleep at night?",
        "On a scale of 1 to 5, how often do you feel motivated to accomplish your daily goals?",
        "On a scale of 1 to 5, how often do you feel overwhelmed by responsibilities?",
        "On a scale of 1 to 5, how often do you feel content with your life?"
    ]
    
    responses = {}
    for i, question in enumerate(questions, start=1):
        while True:
            try:
                print(f"\nQuestion {i}: {question}")
                response = int(input("Your answer (1-5): "))
                if response < 1 or response > 5:
                    raise ValueError("Answer must be between 1 and 5.")
                responses[f"Question {i}"] = response
                break
            except ValueError as e:
                print(f"Invalid input: {e}")

    # Analyzing responses
    stress_level = responses["Question 1"]
    sleep_quality = responses["Question 2"]
    motivation = responses["Question 3"]
    overwhelm = responses["Question 4"]
    life_contentment = responses["Question 5"]

    analysis = "\nEvaluation Results:\n"
    potential_diagnosis = "\nPotential Diagnosis:\n"

    if stress_level > 3:
        analysis += "- You seem to experience high levels of stress. Consider stress-management techniques like meditation, yoga, or exercise.\n"
        potential_diagnosis += "- High stress levels may indicate Generalized Anxiety Disorder (GAD).\n"
    else:
        analysis += "- Your stress levels seem manageable. Keep it up!\n"

    if sleep_quality < 3:
        analysis += "- Your sleep quality could be improved. Prioritize good sleep hygiene.\n"
        potential_diagnosis += "- Poor sleep quality might be linked to Insomnia or Sleep Disorders.\n"
    else:
        analysis += "- Your sleep quality appears good.\n"

    if motivation < 3:
        analysis += "- You might be feeling less motivated. Reflect on your goals and consider seeking support.\n"
        potential_diagnosis += "- Low motivation levels could be a sign of Depression.\n"
    else:
        analysis += "- You seem motivated to achieve your goals.\n"

    if overwhelm > 3:
        analysis += "- Feeling overwhelmed often can be challenging. Try breaking tasks into smaller steps and setting realistic expectations.\n"
        potential_diagnosis += "- Frequently feeling overwhelmed may indicate Burnout or Anxiety Disorders.\n"
    else:
        analysis += "- You seem to manage your responsibilities well.\n"

    if life_contentment < 3:
        analysis += "- You might be feeling less content with your life. Consider practicing gratitude, self-care, or focusing on activities and relationships that bring you joy.\n"
        potential_diagnosis += "- Low life contentment may be a sign of Depression or Chronic Stress.\n"
    else:
        analysis += "- You seem content with your life.\n"

    print(analysis)
    print(potential_diagnosis)

    # Save results to a file
    results = {
        "responses": responses,
        "analysis": analysis,
        "potential_diagnosis": potential_diagnosis
    }
    with open("evaluation_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nYour responses and analysis have been saved to 'evaluation_results.json'.")

if __name__ == "__main__":
    psychological_evaluation()
