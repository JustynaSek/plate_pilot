import streamlit as st
import os
import sys
from meal_plan_advisor import MealPlanAdvisor
from health_advisor import HealthAdvisor
from llm_config import llm, llm_fine_tuned_gpt4o
from memory_config import conversation_memory_summary, previous_meal_plan_memory
from tools import agent_executor
from retriever import get_global_retriever 

retriever = get_global_retriever()
meal_plan_advisor = MealPlanAdvisor(llm, retriever, conversation_memory_summary, previous_meal_plan_memory, agent_executor)
health_advisor = HealthAdvisor(llm_fine_tuned_gpt4o)

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_meal_plan' not in st.session_state:
        st.session_state['current_meal_plan'] = None
    if 'updated_meal_plan' not in st.session_state:
        st.session_state['updated_meal_plan'] = None
    if 'dessert_recipes' not in st.session_state:
        st.session_state['dessert_recipes'] = None
    if 'show_health_tips' not in st.session_state:
        st.session_state['show_health_tips'] = False
    if 'user_name' not in st.session_state:
        st.session_state['user_name'] = ''
    if 'dietary_restrictions' not in st.session_state:
        st.session_state['dietary_restrictions'] = 'general'
    if 'likes' not in st.session_state:
        st.session_state['likes'] = ''
    if 'dislikes' not in st.session_state:
        st.session_state['dislikes'] = ''

def display_meal_plan_section():
    """Display the meal plan section with input fields and buttons."""
    st.subheader("Meal Plan")
    
    st.session_state['user_name'] = st.text_input("Your Name (Optional):", value=st.session_state['user_name'])
    st.session_state['dietary_restrictions'] = st.selectbox(
        "Dietary Restrictions:", 
        ["general", "diabetes", "heart failure", "low-sodium", "vegetarian"],
        index=["general", "diabetes", "heart failure", "low-sodium", "vegetarian"].index(st.session_state['dietary_restrictions'])
    )
    st.session_state['likes'] = st.text_input("Food preferences (e.g., vegetables, pasta, sushi, spicy):", value=st.session_state['likes'])
    st.session_state['dislikes'] = st.text_input("Any foods to avoid? (e.g., rice, heavy sauces, dairy):", value=st.session_state['dislikes'])

    get_meal_plan_button_label = "Get Another Meal Plan" if st.session_state['current_meal_plan'] else "Get My Meal Plan"
    if st.button(get_meal_plan_button_label):
        if not st.session_state['user_name']:
            st.session_state['user_name'] = "User"
        with st.spinner("Generating your personalized meal plan..."):
            response = meal_plan_advisor.generate_meal_plan(
                st.session_state['user_name'],
                st.session_state['dietary_restrictions'],
                st.session_state['likes'],
                st.session_state['dislikes']
            )
            if response:
                st.session_state['current_meal_plan'] = response
                st.session_state['updated_meal_plan'] = None
                st.session_state['dessert_recipes'] = None
                st.rerun()
            else:
                st.error("Could not generate a recipe.")

    if st.session_state['current_meal_plan']:
        st.markdown("### Current Meal Plan")
        st.markdown(st.session_state['current_meal_plan'])
        
        st.markdown("### Modify Meal Plan")
        modification_request = st.text_input("What would you like to change about this meal plan?")
        if st.button("Update Meal Plan"):
            if modification_request:
                with st.spinner("Updating your meal plan..."):
                    response = meal_plan_advisor.generate_modified_meal_plan(
                        st.session_state['user_name'],
                        modification_request,
                        st.session_state['dietary_restrictions'],
                        st.session_state['likes'],
                        st.session_state['dislikes']
                    )
                    if response:
                        st.session_state['updated_meal_plan'] = response
                        st.rerun()
                    else:
                        st.error("Could not generate an improved recipe.")
        
        if st.session_state['updated_meal_plan']:
            st.markdown("### Updated Meal Plan")
            st.markdown(st.session_state['updated_meal_plan'])

def display_dessert_section():
    """Display the dessert recipes section."""
    st.subheader("Dessert Recipes")
    
    if not st.session_state['dessert_recipes']:
        if st.button("Get Dessert Recipes"):
            with st.spinner("Searching for dessert recipes..."):
                response = meal_plan_advisor.generate_dessert_recipe(
                    st.session_state['dietary_restrictions'],
                    st.session_state['likes'],
                    st.session_state['dislikes']
                )
                if response:
                    st.session_state['dessert_recipes'] = response
                    st.rerun()
                else:
                    st.error("Could not find dessert recipes.")
    else:
        st.markdown(st.session_state['dessert_recipes'])
        if st.button("Get Different Dessert Recipes"):
            st.session_state['dessert_recipes'] = None
            st.rerun()

def display_health_tips_section():
    """Display the health tips section."""
    st.subheader("Health Tips")
    
    if not st.session_state['show_health_tips']:
        if st.button("Get Health Tips"):
            with st.spinner("Generating health tips..."):
                response = health_advisor.generate_health_tips_gpt4o(
                    st.session_state['user_name'],
                    st.session_state['dietary_restrictions']
                )
                if response:
                    st.session_state['health_tips'] = response
                    st.session_state['show_health_tips'] = True
                    st.rerun()
                else:
                    st.error("Could not generate health tips.")
    else:
        st.markdown(st.session_state['health_tips'])
        if st.button("Get New Health Tips"):
            st.session_state['show_health_tips'] = False
            st.rerun()

def main():
    st.title("PlatePilot - Your AI Meal Plan Assistant")
    st.subheader("Personalized Meal Plans for You")

    initialize_session_state()

    display_meal_plan_section()
    display_dessert_section()
    display_health_tips_section()

if __name__ == "__main__":
    main()