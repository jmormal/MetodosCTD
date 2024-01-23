import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
from MChains import MarkovChain
import ipywidgets as widgets
# module from this repository


# Input widgets

# Widget for input text to enter the number of states of a Markov chain
states_input = widgets.IntText(
    value=2,
    description='Number of States:',
    style={'description_width': 'initial'}
)


# Update State Names
update_states_name_button = widgets.Button(description="Update State Names")

# Function to handle the button click event
def on_update_states_name_button_clicked(b):
    global matrix_input

    matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
    state_names = [child for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
    # get unique state names
    state_names1 =  state_names[:len(state_names)//2]
    # chage rest of the state names
    for i in range(len(state_names1)):
        state_names[i + len(state_names1)].value = state_names[i].value 


update_states_name_button.on_click(on_update_states_name_button_clicked)

# Set Matrix to 0
set_matrix_to_zero_button = widgets.Button(description="Set Matrix to 0")

# Function to handle the button click event
def on_set_matrix_to_zero_button_clicked(b):
    global matrix_input

    matrix_values = [child for child in matrix_input.children if isinstance(child, widgets.FloatText)]
    for i in range(len(matrix_values)):
        matrix_values[i].value = 0

set_matrix_to_zero_button.on_click(on_set_matrix_to_zero_button_clicked)



# Function to create an n*n matrix input widget with state name inputs in both row and column
def create_matrix_input(n):
    matrix_children = [widgets.Text(value='', disabled=True)]  # Empty top-left corner cell
    # Top row state names
    a=[widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')) for i in range(n)]
    matrix_children.extend([widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')) for i in range(n)])

    for i in range(n):
        # Left column state names
        matrix_children.append(widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')))
        # Row for transition matrix
        matrix_children.extend([widgets.FloatText(value=1/n, step=0.1, layout=widgets.Layout(width='auto')) for _ in range(n)])

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
    matrix_widget.children = base_boxes +[output]



states_input.observe(update_matrix_input, names='value')


# Buttons

# Button to find the steady state probabilities
calc_button = widgets.Button(description="Calculate Steady States Probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'),)
draw_button = widgets.Button(description="Draw Graph", layout=widgets.Layout(flex='1 1 0%', width='auto'))
draw_prob_button = widgets.Button(description="Probability first-passage time from state i to state j ", layout=widgets.Layout(flex='1 1 0%', width='auto'))
calc_first_time_mean_button = widgets.Button(description="Estimated first time passage", layout=widgets.Layout(flex='1 1 0%', width='auto'))
estimate_and_draw_first_time_prob_button = widgets.Button(description="Estimate and draw  first time probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'))
estimate_and_draw_first_time_mean_button = widgets.Button(description="Estimate and draw  first time mean", layout=widgets.Layout(flex='1 1 0%', width='auto'))
calc_matrix_power_button = widgets.Button(description="Calculate Matrix Power", layout=widgets.Layout(flex='1 1 0%', width='auto'))
classify_button = widgets.Button(description="Classify the Markov Chain", layout=widgets.Layout(flex='1 1 0%', width='auto'))

# Button to find the probability of going from state i to state j in n steps 
solve_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_estimated_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_estimate_and_draw_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_estimated_mean_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
solve_matrix_power_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))


# Output widget
output = widgets.Output()
def on_calc_matrix_power_button_clicked(b):


    global matrix_input


    # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, draw_prob_button, i_input, j_input, n_input,solve_prob_button, output]

    matrix_widget.children= base_boxes +[ n_input,solve_matrix_power_button, output]

calc_matrix_power_button.on_click(on_calc_matrix_power_button_clicked)

def on_solve_matrix_power_button_clicked(b):
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
            Mn=M.get_transition_matrix_n_steps(n_input.value)
            # turn to the dataframe
            df = pd.DataFrame(Mn, columns=state_names, index=state_names)
            display(df)
    except Exception as e:
        print(f"Error: {e}")

solve_matrix_power_button.on_click(on_solve_matrix_power_button_clicked)

# Functions to handle the button click events
def on_classify_button_clicked(b):
    with output:
        clear_output()
        n = states_input.value
        matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        state_names =  state_names[:len(state_names)//2]
        transition_matrix = np.array(matrix_values).reshape(n, n)
        try:
            M = MarkovChain(transition_matrix, state_names)
            display( f"The Markov chain is {M.check_reducibility()}")
        except Exception as e:
            display(f"Error: {e}")

classify_button.on_click(on_classify_button_clicked)

# Function to handle the button click event for drawing the graph



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
            for i in range(len(sol)):
                # use a precision of 3 decimal places
                display(f"The steady state probability of {state_names[i]} is {sol[i]:.3f}")
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
            if M.check_if_probabilities():
                M.get_graph()
            else:
                display("Error: Not a valid transition matrix")
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

    matrix_widget.children= base_boxes +[i_input, j_input, n_input,solve_prob_button, output]



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
            # Get the index of the state names
            i = state_names.index(i_input.value)
            j = state_names.index(j_input.value)
            # display(M.get_probability_first_time_passage_n_steps(n_input.value,i,j))
            display(f"The probability of going from state {i_input.value} to state {j_input.value} in {n_input.value} steps is {M.get_probability_first_time_passage_n_steps(n_input.value,i,j)}")
    except Exception as e:
        print(f"Error: {e}")

box_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    height='80px')



items_0 = [calc_button,draw_button,  draw_prob_button, calc_first_time_mean_button , estimate_and_draw_first_time_prob_button , solve_estimated_prob_button]
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
    
items_calc = [calc_button,  draw_prob_button, calc_first_time_mean_button , calc_matrix_power_button]

for item in items_calc:
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
    

box_calc = widgets.Box(children=items_calc, layout=box_layout)


items_estimate = [estimate_and_draw_first_time_prob_button , solve_estimated_prob_button]

for item in items_estimate:
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

box_estimate = widgets.Box(children=items_estimate, layout=box_layout)

items_2 = [classify_button, draw_button]


for item in items_2:
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

box_2 = widgets.Box(children=items_2, layout=box_layout)

box_0 = widgets.Box(children=items_0, layout=box_layout)

items_1=[states_input, update_states_name_button,set_matrix_to_zero_button]
box_1 = widgets.Box(children=items_1, layout=box_layout)

# Function to handle the button click event for calculating first time probabilities
def on_calc_first_time_mean_button_clicked(b):
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
            display(M.draw_probability_distribution_first_time_n_simulation(i,j,n_input.value))
            display("The graph is drawn below: ")

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

    matrix_widget.children= base_boxes +[i_input, j_input, n_input, solve_estimate_and_draw_button, output]


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
            M = MarkovChain(transition_matrix, state_names)
            # Get the index of the state names
            i = state_names.index(i_input.value)
            
            j = state_names.index(j_input.value)
            display(M.estimate_probability_first_time_passage_n_steps(n_input.value,i,j))
    except Exception as e:
        print(f"Error: {e}")
        

# Create button to save output
save_button = widgets.Button(description="Save Output", layout=widgets.Layout(flex='1 1 0%', width='auto'),)
import subprocess
# Function to handle the button click event
def on_save_button_clicked(b):
    # save the output to a pdf file
    print(display.__dict__)
save_button.on_click(on_save_button_clicked)


calc_button.on_click(on_calc_button_clicked)
draw_button.on_click(on_draw_button_clicked)
draw_prob_button.on_click(on_draw_prob_button_clicked)
solve_prob_button.on_click(solve_prob_button_clicked)
solve_estimate_and_draw_button.on_click(solve_estimate_and_draw_first_time_prob_button_clicked)
# Attach the functions to the buttons
calc_first_time_mean_button.on_click(on_calc_first_time_mean_button_clicked)
estimate_and_draw_first_time_prob_button.on_click(on_estimate_and_draw_first_time_prob_button_clicked)
solve_estimated_prob_button.on_click(on_solve_estimated_prob_button_clicked)
# Layout the widgets
base_boxes=[box_2, box_calc, box_estimate,  box_1, matrix_input]
matrix_widget = widgets.VBox(base_boxes +[output])
display(matrix_widget)