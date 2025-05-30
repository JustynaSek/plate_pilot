import streamlit as st
import sys
import os

from langchain_logic.core.meal_plan_advisor import MealPlanAdvisor
from langchain_logic.core.health_advisor import HealthAdvisor
from langchain_logic.core.llm_config import llm, llm_fine_tuned_gpt4o 
from langchain_logic.core.memory_config import conversation_memory_summary, previous_meal_plan_memory
from langchain_logic.core.tools import agent_executor
from langchain_logic.core.retriever import retriever

meal_plan_advisor = MealPlanAdvisor(llm, retriever, conversation_memory_summary, previous_meal_plan_memory, agent_executor)
health_advisor = HealthAdvisor(llm_fine_tuned_gpt4o)

def main():
    st.title("PlatePilot - Your AI Meal Plan Assistant")
    st.subheader("Personalized Meal Plans for You")

    # Initialize session state for persistence
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
            st.session_state['dessert_recipes'] = None # Clear previous desserts
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
                    st.session_state['dessert_recipes'] = None # Clear previous desserts after modifying main plan
                    st.rerun()
                else:
                    st.error("Could not generate an improved recipe.")

    # "Get Dessert Recipes" button is now independent
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
    col1, col2 = st.columns(2) # Use columns for side-by-side buttons

    with col1:
        if st.button("Get Health Tips (GPT-4o Fine-tuned)"):
            with st.spinner("Generating tips with GPT-4o..."):
                response = health_advisor.generate_health_tips(user_name, dietary_restrictions, use_qwen=False)
                if response:
                    st.write(response)
                else:
                    st.error("Could not generate health tips from GPT-4o.")

    with col2:
        # Only show Qwen button if model loaded successfully
        from langchain_logic.core.llm_config import llm_fine_tuned_qwen # Import to check if loaded
        if llm_fine_tuned_qwen: # Check if the Qwen LLM was successfully loaded
            if st.button("Get Health Tips (Qwen Fine-tuned)"):
                with st.spinner("Generating tips with Qwen..."):
                    response = health_advisor.generate_health_tips(user_name, dietary_restrictions, use_qwen=True)
                    if response:
                        st.write(response)
                    else:
                        st.error("Could not generate health tips from Qwen.")
        else:
            st.warning("Qwen Fine-tuned Model is not available.")


    if st.session_state['dessert_recipes']:
        st.subheader("Dessert Recipes:")
        st.write(st.session_state['dessert_recipes'])

if __name__ == "__main__":
    main()