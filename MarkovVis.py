import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
from MChains import MarkovChain

class MarkovChainVisualizer:
    def __init__(self):
        self.setup_widgets()
        self.setup_event_handlers()
        # self.setup_layout()
        self.create_matrix_input(2)
        self.states_input.observe(self.update_matrix_input, names='value')
        matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        # get unique state names
        state_names =  state_names[:len(state_names)//2]

        # Create three input widgets for a sub query
        self.i_input = widgets.Dropdown(
            options=state_names,
            description='From:',
            style={'description_width': 'initial'}
        )
        self.j_input = widgets.Dropdown(
            options=state_names,
            description='To:',
            style={'description_width': 'initial'}
        )
        self.n_input = widgets.IntText(
            value=2,
            description='Number of Steps:',
                style={'description_width': 'initial'})
        
        

        self.box_layout = widgets.Layout(display='flex',
                            flex_flow='row',
                            align_items='stretch',
                            height='80px')



        self.items_0 = [self.calc_button,self.draw_button,  self.draw_prob_button, self.calc_first_time_mean_button , self.estimate_and_draw_first_time_prob_button , self.solve_estimated_prob_button]
        # change height of the buttons to 'auto' to let the button grow vertically
        for item in self.items_0:
            item.layout.height = 'auto'
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
            
        self.items_calc = [self.calc_button,  self.calc_first_time_mean_button, self.calc_matrix_power_button]
        for item in self.items_calc:
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
            

        self.box_calc = widgets.Box(children=self.items_calc, layout=self.box_layout)


        self.items_estimate = [self.estimate_and_draw_first_time_prob_button , self.draw_prob_button,]

        for item in self.items_estimate:
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

        self.box_estimate = widgets.Box(children=self.items_estimate, layout=self.box_layout)

        self.items_2 = [self.classify_button, self.draw_button]


        for item in self.items_2:
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

        self.box_2 = widgets.Box(children=self.items_2, layout=self.box_layout)

        self.box_0 = widgets.Box(children=self.items_0, layout=self.box_layout)

        self.items_1=[self.states_input, self.update_states_name_button,self.set_matrix_to_zero_button]
        self.box_1 = widgets.Box(children=self.items_1, layout=self.box_layout)
        self.base_boxes=[self.box_2, self.box_calc, self.box_estimate,  self.box_1, self.matrix_input]
        self.matrix_widget = widgets.VBox(self.base_boxes +[self.output])
    def setup_widgets(self):
        self.states_input = widgets.IntText(
            value=2,
            description='Number of States:',
            style={'description_width': 'initial'}
        )
        
        self.update_states_name_button = widgets.Button(description="Update State Names")
        self.set_matrix_to_zero_button = widgets.Button(description="Set Matrix to 0")


        # Create other necessary widgets like Dropdowns, IntText, etc.
        self.calc_button = widgets.Button(description="Calculate Steady States Probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'),)
        self.draw_button = widgets.Button(description="Draw Graph", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.draw_prob_button = widgets.Button(description="Probability first-passage time from state i to state j ", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.calc_first_time_mean_button = widgets.Button(description="Estimated first time passage", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.estimate_and_draw_first_time_prob_button = widgets.Button(description="Estimate and draw  first time probabilities", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.estimate_and_draw_first_time_mean_button = widgets.Button(description="Estimate and draw  first time mean", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.calc_matrix_power_button = widgets.Button(description="Calculate Matrix Power", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.classify_button = widgets.Button(description="Classify the Markov Chain", layout=widgets.Layout(flex='1 1 0%', width='auto'))

        # Button to find the probability of going from state i to state j in n steps 
        self.solve_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.solve_estimated_prob_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.solve_estimate_and_draw_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.solve_estimated_mean_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.solve_matrix_power_button = widgets.Button(description="Solve", layout=widgets.Layout(flex='1 1 0%', width='auto'))
        self.set_matrix_to_zero_button = widgets.Button(description="Set Matrix to 0")

        self.output = widgets.Output()

    def setup_event_handlers(self):
        self.update_states_name_button.on_click(self.on_update_states_name_button_clicked)
        self.set_matrix_to_zero_button.on_click(self.on_set_matrix_to_zero_button_clicked)
        self.calc_matrix_power_button.on_click(self.on_calc_matrix_power_button_clicked)
        self.solve_matrix_power_button.on_click(self.on_solve_matrix_power_button_clicked)
        self.classify_button.on_click(self.on_classify_button_clicked)

        #
        # self.save_button.on_click(self.on_save_button_clicked)


        self.calc_button.on_click(self.on_calc_button_clicked)
        self.draw_button.on_click(self.on_draw_button_clicked)
        self.draw_prob_button.on_click(self.on_draw_prob_button_clicked)
        self.solve_prob_button.on_click(self.solve_prob_button_clicked)
        self.solve_estimate_and_draw_button.on_click(self.solve_estimate_and_draw_first_time_prob_button_clicked)
        # Attach the functions to the buttons
        self.calc_first_time_mean_button.on_click(self.on_calc_first_time_mean_button_clicked)
        self.estimate_and_draw_first_time_prob_button.on_click(self.on_estimate_and_draw_first_time_prob_button_clicked)
        self.solve_estimated_prob_button.on_click(self.on_solve_estimated_prob_button_clicked)


# Function to handle the button click event
    def on_update_states_name_button_clicked(self,b):

        matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        # get unique state names
        state_names1 =  state_names[:len(state_names)//2]
        # chage rest of the state names
        for i in range(len(state_names1)):
            state_names[i + len(state_names1)].value = state_names[i].value 

    def on_set_matrix_to_zero_button_clicked(self,b):

        matrix_values = [child for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
        for i in range(len(matrix_values)):
            matrix_values[i].value = 0
        
    def display(self):
        # Display the entire widget
        display(self.matrix_widget)

    def create_matrix_input(self,n):
        matrix_children = [widgets.Text(value='', disabled=True)]  # Empty top-left corner cell
        # Top row state names
        a=[widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')) for i in range(n)]
        matrix_children.extend([widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')) for i in range(n)])

        for i in range(n):
            # Left column state names
            matrix_children.append(widgets.Text(value=f'State {i+1}', layout=widgets.Layout(width='auto')))
            # Row for transition matrix
            matrix_children.extend([widgets.FloatText(value=1/n, step=0.1, layout=widgets.Layout(width='auto')) for _ in range(n)])

        self.matrix_input = widgets.GridBox(
            children=matrix_children,
            layout=widgets.Layout(
                width='100%',
                grid_template_columns=f'auto {" ".join(["auto"] * n)}'
            )
        )
    def update_matrix_input(self,change):
        with self.output:
            clear_output()
        n = change['new']
        self.create_matrix_input(n)
        # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, output]
        self.base_boxes=[self.box_2, self.box_calc, self.box_estimate,  self.box_1, self.matrix_input]
        self.matrix_widget.children = self.base_boxes +[self.output]

    def on_calc_matrix_power_button_clicked(self, b):
        self.matrix_widget.children= self.base_boxes +[ self.n_input,self.solve_matrix_power_button, self.output]


    
    def on_solve_matrix_power_button_clicked(self,b):
        try:
            with self.output:
                clear_output()
                matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names
                state_names =  state_names[:len(state_names)//2]
                n = len(state_names)
                transition_matrix = np.array(matrix_values).reshape(n, n)
                M = MarkovChain(transition_matrix, state_names)
                # Get the index of the state names
                Mn=M.get_transition_matrix_n_steps(self.n_input.value)
                # turn to the dataframe
                df = pd.DataFrame(Mn, columns=state_names, index=state_names)
                display("The transition probability matrix for n steps is")
                display(df)
        except Exception as e:
            print(f"Error: {e}")

    def on_classify_button_clicked(self,b):
        with self.output:
            clear_output()
            n = self.states_input.value
            matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            state_names =  state_names[:len(state_names)//2]
            transition_matrix = np.array(matrix_values).reshape(n, n)
            try:
                M = MarkovChain(transition_matrix, state_names)
                if M.check_if_probabilities():
                    pass
                else:
                    display("Error: Not a valid transition matrix")
                    return 0
            except:
                pass
            try:
                M = MarkovChain(transition_matrix, state_names)
                display( f"The Markov chain is {M.check_reducibility()}")
            except Exception as e:
                display(f"Error: {e}")
    
    def on_calc_button_clicked(self,b):
        with self.output:
            
            clear_output()
            n = self.states_input.value
            matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
            state_names =  state_names[:len(state_names)//2]

            transition_matrix = np.array(matrix_values).reshape(n, n)
            try:
                M = MarkovChain(transition_matrix, state_names)
                if M.check_if_probabilities():
                    pass
                else:
                    display("Error: Not a valid transition matrix")
                    return 0
            except:
                pass

            try:
                M = MarkovChain(transition_matrix, state_names)
                sol = M.get_steady_state()
                for i in range(len(sol)):
                    # use a precision of 3 decimal places
                    display(f"The steady state probability of {state_names[i]} is {sol[i]:.3f}")
                return sol
            except:
                display("The matrix is not reducable") 
    
    def on_draw_button_clicked(self,b):
        with self.output:
            clear_output()
            n = self.states_input.value
            matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
            state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
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


    def on_draw_prob_button_clicked(self,b):
        n = self.states_input.value

        matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
        state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
        # get unique state names
        state_names =  state_names[:len(state_names)//2]

        # set i_input and j_input options
        self.i_input.options = state_names
        self.j_input.options = state_names


        # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, draw_prob_button, i_input, j_input, n_input,solve_prob_button, output]

        self.matrix_widget.children= self.base_boxes +[self.i_input, self.j_input, self.n_input,self.solve_prob_button, self.output]



    def solve_prob_button_clicked(self, b):
        try:
            with self.output:
                clear_output()
                matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names
                state_names =  state_names[:len(state_names)//2]
                n = len(state_names)
                transition_matrix = np.array(matrix_values).reshape(n, n)
                try:
                    M = MarkovChain(transition_matrix, state_names)
                    if M.check_if_probabilities():
                        pass
                    else:
                        display("Error: Not a valid transition matrix")
                        return 0
                except:
                    pass                
                M = MarkovChain(transition_matrix, state_names)
                # Get the index of the state names
                i = state_names.index(self.i_input.value)
                j = state_names.index(self.j_input.value)
                # display(M.get_probability_first_time_passage_n_steps(n_input.value,i,j))
                display(f"The probability of first-time passage from state {self.i_input.value} to state {self.j_input.value} in {self.n_input.value} steps is {M.get_probability_first_time_passage_n_steps(self.n_input.value,i,j)}")
        except Exception as e:
            print(f"Error: {e}")

    # Todo 
    # Function to handle the button click event for calculating first time probabilities
    def on_calc_first_time_mean_button_clicked(self, b):
        try:
            with self.output:
                clear_output()
                matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names    
                state_names =  state_names[:len(state_names)//2]

                n = len(state_names)
                transition_matrix = np.array(matrix_values).reshape(n, n)
                try:
                    M = MarkovChain(transition_matrix, state_names)
                    if M.check_if_probabilities():
                        pass
                    else:
                        display("Error: Not a valid transition matrix")
                        return 0
                except:
                    pass                

                M = MarkovChain(transition_matrix, state_names)

                
                df = pd.DataFrame(M.get_estimated_first_passage_times(), columns=state_names, index=state_names)
                display("The mean first-passage time matrix is:")
                display(df)
                
        except:
            display("The matrix is not reducable") 
    

    # Function to handle the button click event for estimating and drawing first time probabilities
    def solve_estimate_and_draw_first_time_prob_button_clicked(self,b):
        try:
            with self.output:
                clear_output()
                matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names
                state_names =  state_names[:len(state_names)//2]
                n = len(state_names)
                transition_matrix = np.array(matrix_values).reshape(n, n)
                try:
                    M = MarkovChain(transition_matrix, state_names)
                    if M.check_if_probabilities():
                        pass
                    else:
                        display("Error: Not a valid transition matrix")
                        return 0
                except:
                    pass

                M = MarkovChain(transition_matrix, state_names)
                # Get the index of the state names
                i = state_names.index(self.i_input.value)
                j = state_names.index(self.j_input.value)

                a = M.draw_probability_distribution_first_time_n_simulation(i,j,self.n_input.value)
                display(f"The estimated mean first-passage time {a}")
        except Exception as e:
            print(f"Error: {e}")

    def on_estimate_and_draw_first_time_prob_button_clicked(self,b):
        try:
            with self.output:
                n = self.states_input.value
                global matrix_input

                matrix_values = [child.value for child in self.matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in self.matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names
                state_names =  state_names[:len(state_names)//2]
                transition_matrix = np.array(matrix_values).reshape(n, n)
                try:
                    M = MarkovChain(transition_matrix, state_names)
                    if M.check_if_probabilities():
                        pass
                    else:
                        display("Error: Not a valid transition matrix")
                        return 0
                except:
                    pass
                # set i_input and j_input options
                self.i_input.options = state_names
                self.j_input.options = state_names


                # matrix_widget.children = [states_input, matrix_input, calc_button, draw_button, draw_prob_button, i_input, j_input, n_input,solve_prob_button, output]

                self.matrix_widget.children= self.base_boxes +[self.i_input, self.j_input, self.n_input, self.solve_estimate_and_draw_button, self.output]
        except Exception as e:
            print(f"Error: {e}")

    # Function to handle the button click event for solving estimated probabilities
    def on_solve_estimated_prob_button_clicked(self, b):
        try:
            with self.output:
                clear_output()
                matrix_values = [child.value for child in matrix_input.children if isinstance(child, widgets.FloatText)]
                state_names = [child.value for child in matrix_input.children if isinstance(child, widgets.Text) and child.value != '']
                # get unique state names
                state_names =  state_names[:len(state_names)//2]
                n = len(state_names)
                transition_matrix = np.array(matrix_values).reshape(n, n)
                M = MarkovChain(transition_matrix, state_names)
                # Get the index of the state names
                i = state_names.index(self.i_input.value)
                
                j = state_names.index(self.j_input.value)
                display(M.estimate_probability_first_time_passage_n_steps(slef.n_input.value,i,j))
        except Exception as e:
            print(f"Error: {e}")
            

if __name__ == '__main__':
    mc = MarkovChainVisualizer()
    mc.display()
