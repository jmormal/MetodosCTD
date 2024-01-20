import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
from MChains import MarkovChain
# module from this repository


# Input widgets

# Widget for input text to enter the number of states of a Markov chain
states_input = widgets.IntText(
    value=2,
    description='Number of States:',
    style={'description_width': 'initial'}
)




# Function to create an n*n matrix input widget with state name inputs in both row and column
def create_matrix_input(n):
    matrix_children = [widgets.Text(value='', disabled=True)]  # Empty top-left corner cell
    # Top row state names
    matrix_children.extend([widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')) for i in range(n)])

    for i in range(n):
        # Left column state names
        matrix_children.append(widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')))
        # Row for transition matrix # let the step be 0.1
        matrix_children.extend([widgets.FloatText(value=1/n, step=0.1,  layout=widgets.Layout(width='auto')) for _ in range(n)])

    matrix_input = widgets.GridBox(
        children=matrix_children,
        layout=widgets.Layout(
            width='100%',
            grid_template_columns=f'auto {" ".join(["auto"] * n)}'
        )
    )
    return matrix_input

# Initially, create a 2*2 matrix input widget with state names
matrix_input = create_matrix_input(2)

# Update function for the matrix input widget when the number of states changes
def update_matrix_input(change):
    with output:
        clear_output()
    n = change['new']
    global matrix_input
    matrix_input = create_matrix_input(n)
    # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, output]
    matrix_widget.children = [box_0,states_input, matrix_input, output]



states_input.observe(update_matrix_input, names='value')


# Buttons

# Button to find the steady state probabilities
calc_button = widgets.Button(description="Calculate Steady States", layout=widgets.Layout(flex='1 1 0%', width='auto'),)
draw_button = widgets.Button(description="Draw Graph", layout=widgets.Layout(flex='1 1 0%', width='auto'))
draw_prob_button = widgets.Button(description="Calculate probabilitye \n fsd", layout=widgets.Layout(flex='1 1 0%', width='auto'))
calc_first_time_prob_button = widgets.Button(description="Calculate first time probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'))
estimate_and_draw_first_time_prob_button = widgets.Button(description="Estimate and draw    \n first time probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'))



# Button to find the probability of going from state i to state j in n steps 
solve_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_estimated_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_estimate_and_draw_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))




# Output widget
output = widgets.Output()


# Functions to handle the button click events

# Function to handle the button click event
def on_calc_button_clicked(b):
    with output:
        clear_output()
        n = states_input.value
        matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        state_names =  state_names[:len(state_names)//2]

        transition_matrix = np.array(matrix_values).reshape(n, n)
        try:
            M = MarkovChain(transition_matrix, state_names)
            sol = M.get_steady_state()
            display(f"Steady States: {sol}")
            return sol
        except Exception as e:
            display(f"Error: {e}")

def on_draw_button_clicked(b):
    with output:
        clear_output()
        n = states_input.value
        matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        # get unique state names
        state_names =  state_names[:len(state_names)//2]
        transition_matrix = np.array(matrix_values).reshape(n, n)
        try:
            M = MarkovChain(transition_matrix, state_names)
            M.get_graph()
        except Exception as e:
            display(f"Error: {e}")
n = states_input.value

matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
# get unique state names
state_names =  state_names[:len(state_names)//2]
transition_matrix = np.array(matrix_values).reshape(n, n)

# Create three input widgets for a sub query
i_input = widgets.Dropdown(
    options=state_names,
    description='From:',
    style={'description_width': 'initial'}
)
j_input = widgets.Dropdown(
    options=state_names,
    description='To:',
    style={'description_width': 'initial'}
)
n_input = widgets.IntText(
    value=2,
    description='Number of Steps:',
        style={'description_width': 'initial'})
def on_draw_prob_button_clicked(b):
    n = states_input.value
    global matrix_input

    matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
    state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
    # get unique state names
    state_names =  state_names[:len(state_names)//2]

    transition_matrix = np.array(matrix_values).reshape(n, n)
    # set i_input and j_input options
    i_input.options = state_names
    j_input.options = state_names


    # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, draw_prob_button, i_input, j_input, n_input,solve_prob_button, output]

    matrix_widget.children= [box_0, states_input, matrix_input, i_input, j_input, n_input,solve_prob_button, output]



def solve_prob_button_clicked(b):
    try:
        with output:
            clear_output()
            matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            # get unique state names
            state_names =  state_names[:len(state_names)//2]
            n = len(state_names)
            transition_matrix = np.array(matrix_values).reshape(n, n)
            M = MarkovChain(transition_matrix, state_names)
            display(i_input.value)
            display(j_input.value)
            display(n_input.value)
            # Get the index of the state names
            i = state_names.index(i_input.value)
            j = state_names.index(j_input.value)
            display(M.get_probability_first_time_passage_n_steps(n_input.value,i,j))
    except Exception as e:
        print(f"Error: {e}")

box_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    height='80px')



items_0 = [calc_button,draw_button,  draw_prob_button, calc_first_time_prob_button , estimate_and_draw_first_time_prob_button , solve_estimated_prob_button]
# change height of the buttons to 'auto' to let the button grow vertically
for item in items_0:
    item.layout.height = 'auto'

    
for item in items_0:
    item.style.button_color = 'lightblue'
    item.style.font_weight = 'bold'
    item.style.font_size = 'large'
    item.style.color = 'black'
    item.style.border_color = 'black'
    item.style.border_width = '1px'
    item.style.border_radius = '5px'
    item.style.padding = '10px'
    item.style.margin = '10px'
    item.layout.width = 'auto'
    item.layout.height = 'auto'
    item.layout.flex = '1 1 0%'
    item.layout.align_items = 'stretch'
    item.layout.justify_content = 'center'
    item.layout.align_content = 'center'
    item.layout.display = 'flex'
    item.layout.flex_flow = 'row'
    item.layout.flex_wrap = 'wrap'
    



box_0 = widgets.Box(children=items_0, layout=box_layout)


# Function to handle the button click event for calculating first time probabilities
def on_calc_first_time_prob_button_clicked(b):
    try:
        with output:
            clear_output()
            matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            # get unique state names    
            state_names =  state_names[:len(state_names)//2]

            n = len(state_names)
            transition_matrix = np.array(matrix_values).reshape(n, n)
            print(transition_matrix)
            M = MarkovChain(transition_matrix, state_names)

            
            df = pd.DataFrame(M.get_estimated_first_passage_times(), columns=state_names, index=state_names)
            display(df)
            
    except Exception as e:
        print(f"Error: {e}")

# Function to handle the button click event for estimating and drawing first time probabilities
def solve_estimate_and_draw_first_time_prob_button_clicked(b):
    try:
        with output:
            clear_output()
            matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            # get unique state names
            state_names =  state_names[:len(state_names)//2]
            n = len(state_names)
            transition_matrix = np.array(matrix_values).reshape(n, n)
            M = MarkovChain(transition_matrix, state_names)
            # Get the index of the state names
            i = state_names.index(i_input.value)
            j = state_names.index(j_input.value)
            M.draw_probability_distribution_first_time_n_simulation(i,j,n_input.value)
    except Exception as e:
        print(f"Error: {e}")

def on_estimate_and_draw_first_time_prob_button_clicked(b):
    n = states_input.value
    global matrix_input

    matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
    state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
    # get unique state names
    state_names =  state_names[:len(state_names)//2]
    transition_matrix = np.array(matrix_values).reshape(n, n)
    # set i_input and j_input options
    i_input.options = state_names
    j_input.options = state_names


    # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, draw_prob_button, i_input, j_input, n_input,solve_prob_button, output]

    matrix_widget.children= [box_0, states_input, matrix_input, i_input, j_input, n_input, solve_estimate_and_draw_button, output]


# Function to handle the button click event for solving estimated probabilities
def on_solve_estimated_prob_button_clicked(b):
    try:
        with output:
            clear_output()
            matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            # get unique state names
            state_names =  state_names[:len(state_names)//2]
            n = len(state_names)
            transition_matrix = np.array(matrix_values).reshape(n, n)
            print(transition_matrix)
            M = MarkovChain(transition_matrix, state_names)
            # Get the index of the state names
            i = state_names.index(i_input.value)
            
            j = state_names.index(j_input.value)
            M.solve_estimated_probabilities(n_input.value,i,j)
    except Exception as e:
        print(f"Error: {e}")
        
calc_button.on_click(on_calc_button_clicked)
draw_button.on_click(on_draw_button_clicked)
draw_prob_button.on_click(on_draw_prob_button_clicked)
solve_prob_button.on_click(solve_prob_button_clicked)
solve_estimate_and_draw_button.on_click(solve_estimate_and_draw_first_time_prob_button_clicked)
# Attach the functions to the buttons
calc_first_time_prob_button.on_click(on_calc_first_time_prob_button_clicked)
estimate_and_draw_first_time_prob_button.on_click(on_estimate_and_draw_first_time_prob_button_clicked)
solve_estimated_prob_button.on_click(on_solve_estimated_prob_button_clicked)
# Layout the widgets
matrix_widget = widgets.VBox([box_0, states_input, matrix_input,   output])
display(matrix_widget)