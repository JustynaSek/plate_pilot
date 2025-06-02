# src/streamlit_app.py
import streamlit as st
import os

print("DEBUG streamlit_app: Starting app script execution.") # DEBUG print

from meal_plan_advisor import MealPlanAdvisor
from health_advisor import HealthAdvisor
from llm_config import llm, llm_fine_tuned_gpt4o, llm_fine_tuned_qwen, qwen_tokenizer 
from memory_config import conversation_memory_summary, previous_meal_plan_memory
from tools import agent_executor
from retriever import retriever

print("DEBUG streamlit_app: All module imports successful.") # DEBUG print

print("DEBUG streamlit_app: Initializing MealPlanAdvisor...") # DEBUG print
meal_plan_advisor = MealPlanAdvisor(llm, retriever, conversation_memory_summary, previous_meal_plan_memory, agent_executor)
print("DEBUG streamlit_app: MealPlanAdvisor initialized.") # DEBUG print

print("DEBUG streamlit_app: Initializing HealthAdvisor...") # DEBUG print
health_advisor = HealthAdvisor(llm_fine_tuned_gpt4o, llm_fine_tuned_qwen) # HealthAdvisor already prints debug msgs inside
print("DEBUG streamlit_app: HealthAdvisor initialized.") # DEBUG print

def main():
    print("DEBUG streamlit_app: main() function started. Setting up UI.") # DEBUG print
    st.title("PlatePilot - Your AI Meal Plan Assistant")
    st.subheader("Personalized Meal Plans for You")

    if 'current_meal_plan' not in st.session_state:
        st.session_state['current_meal_plan'] = None
    if 'dessert_recipes' not in st.session_state:
        st.session_state['dessert_recipes'] = None

    user_name = st.text_input("Your Name (Optional):")

    dietary_restrictions = st.selectbox("Dietary Restrictions:", ["general", "diabetes", "heart failure", "low-sodium", "vegetarian"])
    likes = st.text_input("Food preferences (e.g., vegetables, pasta, sushi, spicy):")
    dislikes = st.text_input("Any foods to avoid? (e.g., rice, heavy sauces, dairy):")

    get_meal_plan_button_label = "Get My Meal Plan"
    if st.session_state['current_meal_plan']:
        get_meal_plan_button_label = "Get Another Meal Plan"

    if st.button(get_meal_plan_button_label):
        if not user_name:
            user_name = "User"
        response = meal_plan_advisor.generate_meal_plan(user_name, dietary_restrictions, likes, dislikes)
        if response:
            st.session_state['current_meal_plan'] = response
            st.session_state['dessert_recipes'] = None
        else:
            st.error("Could not generate a recipe.")

    if st.session_state['current_meal_plan']:
        st.subheader("Current Meal Plan:")
        st.write(st.session_state['current_meal_plan'])

        modification_request = st.text_input("What would you like to change about this meal plan?")

        if st.button("Show Meal Plan with your suggestions"):
            if modification_request:
                response = meal_plan_advisor.generate_modified_meal_plan(
                    user_name,
                    modification_request,
                    dietary_restrictions,
                    likes,
                    dislikes
                )
                if response:
                    st.session_state['current_meal_plan'] = response
                    st.session_state['dessert_recipes'] = None
                    st.rerun()
                else:
                    st.error("Could not generate an improved recipe.")

    if st.button("Get Dessert Recipes"):
        with st.spinner("Searching the web for dessert recipes..."):
            response_from_agent = meal_plan_advisor.generate_dessert_recipe(
                dietary_restrictions, likes, dislikes
            )
            if response_from_agent:
                st.session_state['dessert_recipes'] = response_from_agent
            else:
                st.error("Could not find dessert recipes using web search.")

    st.subheader("Health Tips")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Get Health Tips (GPT-4o Fine-tuned)"):
            with st.spinner("Generating tips with GPT-4o..."):
                response = health_advisor.generate_health_tips_gpt4o(user_name, dietary_restrictions)
                if response:
                    st.write(response)
                else:
                    st.error("Could not generate health tips from GPT-4o.")

    with col2:
        if llm_fine_tuned_qwen: 
            if st.button("Get Health Tips (Qwen Fine-tuned)"):
                with st.spinner("Generating tips with Qwen..."):
                    response = health_advisor.generate_health_tips_qwen(user_name, dietary_restrictions)
                    if response:
                        st.write(response)
                    else:
                        st.error("Could not generate health tips from Qwen.")
        else:
            st.warning("Qwen Fine-tuned Model is not available. (Disabled for testing)")


    if st.session_state['dessert_recipes']:
        st.subheader("Dessert Recipes:")
        st.write(st.session_state['dessert_recipes'])

if __name__ == "__main__":
    print("DEBUG streamlit_app: Calling main() function block.") # DEBUG print
    main()
    print("DEBUG streamlit_app: main() function completed. App should be ready.") # DEBUG print