from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, to_hex, to_rgb 
import random
import numpy as np
import pandas as pd
import sknetwork as skn
from sknetwork.visualization import svg_graph, svg_bigraph
from scipy import sparse
import os 
import plotly.graph_objs as go
import math 
import time


output_folder = "model_output"
os.makedirs(output_folder, exist_ok=True)


class Institution(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.property_rights = 10
        self.base_rights_step = 0.5
        self.boost_applied = False
        self.revolutionary_boost = 10
        self.revolutionaries = 0
        self.satisfaction =  0
        self.initial_threshold_difference = 40

    def step(self):
        total_agents = len(self.model.schedule.agents)
        if total_agents == 0:
            return

        self.revolutionaries = sum(1 for agent in self.model.schedule.agents if agent.revolutionaries)
        self.satisfaction = sum(1 for agent in self.model.schedule.agents if agent.satisfaction)
        non_revolutionaries = total_agents - self.revolutionaries
        difference = self.revolutionaries - non_revolutionaries
        self.revolutionary_boost = random.randint(5, 150)
        revolutionary_ratio = self.revolutionaries / total_agents

        if difference > 0:
            adjustment_factor = math.log(difference + 1)
        else:
            adjustment_factor = -math.log(abs(difference) + 1)

        if difference >= self.initial_threshold_difference:
            if not self.boost_applied:
                #self.property_rights += self.revolutionary_boost * (5 * np.log1p(difference))
                self.property_rights += self.revolutionary_boost
                self.boost_applied = True
                print(f"Boost applied! New property rights: {self.property_rights}")
        else:
            if self.boost_applied:
                self.property_rights = max(self.property_rights - 5 * self.base_rights_step, 0)
                self.boost_applied = False
                print(f"Property rights decreased to: {self.property_rights}")

class Innovator(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.influence = 10
        self.knowledge = 10
        self.satisfaction = False
        self.revolutionaries = True
        self.revolution_threshold = random.randint(5, 300)
        self.satisfaction_threshold = random.randint(5, 300)
        self.ideas = set()

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        revolutionary_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'revolutionaries') and neighbor.revolutionaries)
        satisfied_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'satisfaction') and neighbor.satisfaction)

        self.revolutionaries = revolutionary_neighbors >= self.revolution_threshold
        self.satisfaction = satisfied_neighbors >= self.satisfaction_threshold

        if self.revolutionaries:
            self.satisfaction = False
        elif self.satisfaction:
            self.revolutionaries = False

        for neighbor in neighbors:
            if not isinstance(neighbor, Institution):
                neighbor.knowledge = min(max(neighbor.knowledge + self.influence * 0.1, 0), 2000)
                neighbor.ideas.update(self.ideas)
                neighbor.satisfaction_threshold = min(300, max(1, neighbor.satisfaction_threshold - self.knowledge * 0.03 - self.model.institution.property_rights * 0.04))
                neighbor.revolution_threshold = min(300, max(1, neighbor.revolution_threshold - self.knowledge * 0.016 + self.model.institution.property_rights * 0.03))

        self.influence = min(self.influence + 0.0010, 30)


class Noble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 50
        self.satisfaction = True
        self.knowledge = 5
        self.influence = 5
        self.revolutionaries = False
        self.revolution_threshold = random.randint(5, 300)
        self.satisfaction_threshold = random.randint(5, 300)
        self.ideas = set()

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        revolutionary_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'revolutionaries') and neighbor.revolutionaries)
        satisfied_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'satisfaction') and neighbor.satisfaction)

        self.revolutionaries = revolutionary_neighbors >= self.revolution_threshold
        self.satisfaction = satisfied_neighbors >= self.satisfaction_threshold

        if self.revolutionaries:
            self.satisfaction = False
        elif self.satisfaction:
            self.revolutionaries = False

        for neighbor in neighbors:
            if not isinstance(neighbor, Institution):
                neighbor.satisfaction_threshold = min(300, max(1, neighbor.satisfaction_threshold - self.influence * 0.016 - self.model.institution.property_rights * 0.04))
                neighbor.revolution_threshold = min(300, max(1, neighbor.revolution_threshold - self.influence * 0.02 + self.model.institution.property_rights * 0.03))

        self.influence = min(self.influence + 0.001, 30)


class Bourgeois(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 25
        self.satisfaction = False
        self.knowledge = 2
        self.influence = 0
        self.revolutionaries = False
        self.revolution_threshold = random.randint(5, 300)
        self.satisfaction_threshold = random.randint(5, 300)
        self.ideas = set()

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        revolutionary_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'revolutionaries') and neighbor.revolutionaries)
        satisfied_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'satisfaction') and neighbor.satisfaction)

        self.revolutionaries = revolutionary_neighbors >= self.revolution_threshold
        self.satisfaction = satisfied_neighbors >= self.satisfaction_threshold

        # Imposer l'exclusivité
        if self.revolutionaries:
            self.satisfaction = False
        elif self.satisfaction:
            self.revolutionaries = False

        for neighbor in neighbors:
            if not isinstance(neighbor, Institution):
                neighbor.satisfaction_threshold = min(300, max(1, neighbor.satisfaction_threshold - self.knowledge * 0.03 - self.model.institution.property_rights * 0.03))
                neighbor.revolution_threshold = min(300, max(1, neighbor.revolution_threshold - self.knowledge * 0.03 + self.model.institution.property_rights * 0.03))

        self.influence = min(self.influence + 0.001, 30)

class People(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.satisfaction = False
        self.knowledge = 1
        self.influence = 0
        self.revolutionaries = False
        self.revolution_threshold = random.randint(5, 300)
        self.satisfaction_threshold = random.randint(5, 300)
        self.ideas = set()

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        revolutionary_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'revolutionaries') and neighbor.revolutionaries)
        satisfied_neighbors = sum(1 for neighbor in neighbors if hasattr(neighbor, 'satisfaction') and neighbor.satisfaction)

        self.revolutionaries = revolutionary_neighbors >= self.revolution_threshold
        self.satisfaction = satisfied_neighbors >= self.satisfaction_threshold

        if self.revolutionaries:
            self.satisfaction = False
        elif self.satisfaction:
            self.revolutionaries = False

        for neighbor in neighbors:
            if not isinstance(neighbor, Institution):
                neighbor.satisfaction_threshold = min(300, max(1, neighbor.satisfaction_threshold - self.knowledge * 0.03 - self.model.institution.property_rights * 0.03))
                neighbor.revolution_threshold = min(300, max(1, neighbor.revolution_threshold - self.knowledge * 0.03 + self.model.institution.property_rights * 0.03))

        self.influence = min(self.influence + 0.001, 30)

class InstitutionModel(Model):
    def __init__(self, N, G):
        super().__init__()
        self.num_agents = N
        self.grid = NetworkGrid(G)
        self.schedule = RandomActivation(self)
        self.institution = Institution(N, self)
        self.schedule.add(self.institution)

        num_nobles = int(N * 0.04)
        node_positions = list(G.nodes())
        for i, node in enumerate(node_positions):
            if i < num_nobles:
                agent = Noble(i, self)
            elif i < num_nobles + int(N * 0.008):
                agent = Innovator(i, self)
            elif i < num_nobles + int(N * 0.05):
                agent = Bourgeois(i, self)
            else:
                agent = People(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.grid.place_agent(self.institution, node_positions[0])

        for agent in self.schedule.agents:
            if agent != self.institution:
                G.add_edge(self.institution.pos, agent.pos)

        self.datacollector = DataCollector(
            model_reporters={"PropertyRights": lambda m: m.institution.property_rights, "Revolutionaries": lambda m: m.institution.revolutionaries, "Satisfaction": lambda m: m.institution.satisfaction},
            agent_reporters={"Satisfaction": "satisfaction", "knowledge": "knowledge", "satisfaction_threshold": "satisfaction_threshold", "revolution_threshold": "revolution_threshold"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

G = nx.erdos_renyi_graph(150, 0.1)

model = InstitutionModel(1500, G)

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def save_at_step(step):
    data = model.datacollector.get_model_vars_dataframe()

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_size = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        agents = model.grid.get_cell_list_contents([node])
        color = 'black'
        total_influence = max(sum(agent.influence for agent in agents if isinstance(agent, (Innovator, Noble))), 0)

        for agent in agents:
            if isinstance(agent, Noble):
                color = 'lime'
            elif isinstance(agent, Innovator):
                color = 'darkorange'
            elif isinstance(agent, Bourgeois):
                color = 'lightcoral'
            elif isinstance(agent, People):
                color = 'lightblue'
            elif isinstance(agent, Institution):
                color = 'violet'
                
        node_size.append(total_influence*5)
        node_colors.append(to_hex(color))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Total Influence',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Interactive Network Graph at Step {step}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Institution",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    )
                    )

    fig.write_html(f"{output_folder}/network_step_{step}.html")

    # revolutionnaire
    revolutionaries_data = data["Revolutionaries"]
    smoothed_revolutionaries = moving_average(revolutionaries_data, window_size=10)
    
    plt.figure(figsize=(12, 6))
    plt.plot(revolutionaries_data.index, revolutionaries_data.values, label="Nombre de Révolutionnaires (Instantanés)", alpha=0.3)
    plt.plot(revolutionaries_data.index[len(revolutionaries_data) - len(smoothed_revolutionaries):], smoothed_revolutionaries, label="Nombre de Révolutionnaires (Lissé)", color='red')
    plt.title(f"Évolution du Nombre de Révolutionnaires (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Nombre de Révolutionnaires")
    plt.legend()
    plt.savefig(f"{output_folder}/revolutionaries_step_{step}.png")
    plt.close()

    # Graphique des droits de propriété
    property_rights_data = data["PropertyRights"]
    smoothed_property_rights = moving_average(property_rights_data, window_size=10)

    plt.figure(figsize=(12, 6))
    plt.plot(property_rights_data.index, property_rights_data.values, label="Droits de Propriété (Instantanés)", alpha=0.3)
    plt.plot(property_rights_data.index[len(property_rights_data) - len(smoothed_property_rights):], smoothed_property_rights, label="Droits de Propriété (Lissés)", color='blue')
    plt.title(f"Évolution des Droits de Propriété (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Droits de Propriété")
    plt.legend()
    plt.savefig(f"{output_folder}/property_rights_step_{step}.png")
    plt.close()

    # revolution
    agent_data = model.datacollector.get_agent_vars_dataframe()
    revolution_threshold_data = agent_data.groupby("Step").mean()["revolution_threshold"]
    smoothed_revolution_threshold = moving_average(revolution_threshold_data, window_size=10)

    plt.figure(figsize=(12, 6))
    plt.plot(revolution_threshold_data.index, revolution_threshold_data.values, label="Seuil de Révolution (Instantané)", alpha=0.3)
    plt.plot(revolution_threshold_data.index[len(revolution_threshold_data) - len(smoothed_revolution_threshold):], smoothed_revolution_threshold, label="Seuil de Révolution (Lissé)", color='purple')
    plt.title(f"Évolution du Seuil de Révolution (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Seuil de Révolution")
    plt.legend()
    plt.savefig(f"{output_folder}/revolution_threshold_step_{step}.png")
    plt.close()

    # satisfaction threshold
    agent_data = model.datacollector.get_agent_vars_dataframe()
    satisfaction_threshold_data = agent_data.groupby("Step").mean()["satisfaction_threshold"]
    smoothed_satisfaction_threshold = moving_average(satisfaction_threshold_data, window_size=10)

    plt.figure(figsize=(12, 6))
    plt.plot(satisfaction_threshold_data.index, satisfaction_threshold_data.values, label="Seuil de Satisfaction (Instantané)", alpha=0.3)
    plt.plot(satisfaction_threshold_data.index[len(satisfaction_threshold_data) - len(smoothed_satisfaction_threshold):], smoothed_satisfaction_threshold, label="Seuil de Satisfaction (Lissé)", color='green')
    plt.title(f"Évolution du Seuil de Satisfaction (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Seuil de Satisfaction")
    plt.legend()
    plt.savefig(f"{output_folder}/satisfaction_threshold_step_{step}.png")
    plt.close()

    # satisfaction
    satisfaction_data = data["Satisfaction"]
    smoothed_satisfaction = moving_average(satisfaction_data, window_size=10)

    plt.figure(figsize=(12, 6))
    plt.plot(satisfaction_data.index, satisfaction_data.values, label="Satisfaction (Instantané)", alpha=0.3)
    plt.plot(satisfaction_data.index[len(satisfaction_data) - len(smoothed_satisfaction):], smoothed_satisfaction, label="Satisfaction (Lissé)", color='green')
    plt.title(f"Évolution de la Satisfaction (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Satisfaction")
    plt.legend()
    plt.savefig(f"{output_folder}/satisfaction_step_{step}.png")
    plt.close()

    agent_data = model.datacollector.get_agent_vars_dataframe()
    knowledges_data = agent_data.groupby("Step").mean()["knowledge"]
    smoothed_knowledge = moving_average(knowledges_data, window_size=10)
    
    plt.figure(figsize=(12, 6))
    plt.plot(knowledges_data.index, knowledges_data.values, label="Evolution de la connaissance (Instantanés)", alpha=0.3)
    plt.plot(knowledges_data.index[len(knowledges_data) - len(smoothed_knowledge):], smoothed_knowledge, label="Evolution de la connaissance (Lissé)", color='orange')
    plt.title(f"Évolution de la connaissance (Step {step})")
    plt.xlabel("Étape")
    plt.ylabel("Connaissance")
    plt.legend()
    plt.savefig(f"{output_folder}/connaissance_step_{step}.png")
    plt.close()

    #doublette 
    revolutionaries_data = data["Revolutionaries"]
    smoothed_revolutionaries = moving_average(revolutionaries_data, window_size=10)

    satisfaction_data = data["Satisfaction"]
    smoothed_satisfaction = moving_average(satisfaction_data, window_size=10)

    plt.figure(figsize=(12, 6))
    # Plot revolutionaries data
    plt.plot(revolutionaries_data.index, revolutionaries_data.values, label="Révolutionnaires (Instantané)", alpha=0.3)
    plt.plot(revolutionaries_data.index[len(revolutionaries_data) - len(smoothed_revolutionaries):], smoothed_revolutionaries, label="Révolutionnaires (Lissé)", color='red')
    # Plot satisfaction data
    plt.plot(satisfaction_data.index, satisfaction_data.values, label="Satisfait (Instantané)", alpha=0.3)
    plt.plot(satisfaction_data.index[len(satisfaction_data) - len(smoothed_satisfaction):], smoothed_satisfaction, label="Satisfait (Lissé)", color='green')
    plt.title("Évolution des Révolutionnaires et des Satisfaits")
    plt.xlabel("Étape")
    plt.ylabel("Valeur")
    plt.legend()
    plt.savefig(f"{output_folder}/doubletterevosatis_step_{step}.png")
    plt.close()

    #doublette 
    # Assuming agent_data is your DataCollector's agent data
    agent_data = model.datacollector.get_agent_vars_dataframe()

    revolution_threshold_data = agent_data.groupby("Step").mean()["revolution_threshold"]
    smoothed_revolution_threshold = moving_average(revolution_threshold_data, window_size=10)

    satisfaction_threshold_data = agent_data.groupby("Step").mean()["satisfaction_threshold"]
    smoothed_satisfaction_threshold = moving_average(satisfaction_threshold_data, window_size=10)

    plt.figure(figsize=(12, 6))

    # Plot revolution threshold data
    plt.plot(revolution_threshold_data.index, revolution_threshold_data.values, label="Seuil de Révolution (Instantané)", alpha=0.3)
    plt.plot(revolution_threshold_data.index[len(revolution_threshold_data) - len(smoothed_revolution_threshold):], smoothed_revolution_threshold, label="Seuil de Révolution (Lissé)", color='red')

    # Plot satisfaction threshold data
    plt.plot(satisfaction_threshold_data.index, satisfaction_threshold_data.values, label="Seuil de Satisfaction (Instantané)", alpha=0.3)
    plt.plot(satisfaction_threshold_data.index[len(satisfaction_threshold_data) - len(smoothed_satisfaction_threshold):], smoothed_satisfaction_threshold, label="Seuil de Satisfaction (Lissé)", color='green')

    plt.title("Évolution des Seuils de Révolution et de Satisfaction")
    plt.xlabel("Étape")
    plt.ylabel("Valeur")
    plt.legend()
    plt.savefig(f"{output_folder}/doubletteseuil_{step}.png")
    plt.close()

    #triplette 
    revolutionaries_data = data["Revolutionaries"]
    smoothed_revolutionaries = moving_average(revolutionaries_data, window_size=10)

    satisfaction_data = data["Satisfaction"]
    smoothed_satisfaction = moving_average(satisfaction_data, window_size=10)

    property_data = data["PropertyRights"]
    smoothed_property = moving_average(property_data, window_size=10)

    plt.figure(figsize=(12, 6))
    # Plot revolutionaries data
    plt.plot(revolutionaries_data.index, revolutionaries_data.values, label="Révolutionnaires (Instantané)", alpha=0.3)
    plt.plot(revolutionaries_data.index[len(revolutionaries_data) - len(smoothed_revolutionaries):], smoothed_revolutionaries, label="Révolutionnaires (Lissé)", color='red')
    # Plot satisfaction data
    plt.plot(satisfaction_data.index, satisfaction_data.values, label="Satisfait (Instantané)", alpha=0.3)
    plt.plot(satisfaction_data.index[len(satisfaction_data) - len(smoothed_satisfaction):], smoothed_satisfaction, label="Satisfait (Lissé)", color='green')
    plt.plot(property_data.index, property_data.values, label="Droits de propriété (Instantanée)", alpha=0.3)
    plt.plot(property_data.index[len(property_data) - len(smoothed_property):], smoothed_property, label="Droits de propriété (Lissé)", color='blue')
    plt.title("Évolution des Révolutionnaires, des Satisfaits et des droits de propriété")
    plt.xlabel("Étape")
    plt.ylabel("Valeur")
    plt.legend()
    plt.savefig(f"{output_folder}/trilpetterevosatis_step_{step}.png")
    plt.close()


start_time = time.time()

# Run the model and save outputs at specific steps
for i in range(1, 10):
    model.step()
    if i in [10, 50, 100, 150, 200, 250, 500, 600, 700, 800, 900, 999, 1000, 1200, 1500, 1700, 1999, 2500, 2999, 3500, 3999, 4500, 4999, 5500, 5999, 6500, 6999, 7500, 7999, 8500, 8999, 9500, 9999]:
        save_at_step(i)

end_time = time.time()
total_time = end_time - start_time
print(f"Temps total: {total_time} secondes")


data = model.datacollector.get_model_vars_dataframe()
print(data)

print(G.number_of_nodes())
print(G.number_of_edges())

node_colors = []
node_sizes = []

for node in G.nodes():
    agents = model.grid.get_cell_list_contents([node])
    color = 'black'  # Couleur par défaut (noir)

    for agent in agents:
        if isinstance(agent, Noble):
            color = 'lime'  # Vert pastel pour les Nobles
        elif isinstance(agent, Innovator):
            color = 'darkorange'  # Orange pour les Innovators
        elif isinstance(agent, Bourgeois):
            color = 'lightcoral'  # Rouge pastel pour les Bourgeois
        elif isinstance(agent, People):
            color = 'lightblue'  # Bleu pastel pour le People
        elif isinstance(agent, Institution):
            color = 'violet' 

    # Ajouter la couleur au graphe
    node_colors.append(to_hex(color))
    node_sizes.append(G.degree(node))

# Définir la taille minimale et maximale des nœuds
min_size = 10
max_size = 15

# Calculer le degré minimum et maximum
min_degree = min(node_sizes)
max_degree = max(node_sizes)

# Calculer les tailles des nœuds
node_sizes = [(size - min_degree) / (max_degree - min_degree) * (max_size - min_size) + min_size for size in node_sizes]

# Vérifier que la taille des listes est correcte
assert len(node_colors) == len(G.nodes())
assert len(node_sizes) == len(G.nodes())

# Convertir le graphique en matrice d'adjacence
adjacency = nx.to_numpy_array(G)
adjacency = sparse.csr_matrix(adjacency)

# Calculer la taille moyenne des nœuds
avg_size = sum(node_sizes) / len(node_sizes)

# Créer le graphique SVG avec sknetwork
svg = svg_graph(adjacency, node_color=node_colors, node_size=avg_size)

# Enregistrer le graphique en tant que fichier SVG
with open("network.svg", "wb") as f:
    f.write(svg.encode('utf-8'))

print("Graphique enregistré sous 'network.svg'.")


#Fruchterman-Reingold 
pos = nx.spring_layout(G)


edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=[G.degree[node] * 1.5 for node in G.nodes()],  # Taille basée sur le degré du nœud
        color=node_colors,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2
    )
)

node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_info = f'# of connections: {len(adjacencies[1])}'
    node_text.append(node_info)

node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Interactive Network Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    annotations=[dict(
                        text="Institution",
                        showarrow=False,
                        xref="paper", yref="paper"
                    )],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
                )

fig.show()

# Fruchterman-Reingold 
pos = nx.spring_layout(G)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
node_size = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    agents = model.grid.get_cell_list_contents([node])
    total_influence = max(sum(agent.influence for agent in agents if isinstance(agent, (Innovator, Noble))), 0)

    node_size.append(total_influence*5)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=node_size,  # Taille basée sur l'influence totale
        color=node_colors,
        colorbar=dict(
            thickness=15,
            title='Total Influence',
            xanchor='left',
            titleside='right'
        ),
        line_width=2
    )
)

node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    agents = model.grid.get_cell_list_contents([node])
    total_influence = sum(agent.influence for agent in agents if isinstance(agent, (Innovator, Noble)))
    node_info = f'Total Influence: {total_influence}'
    node_text.append(node_info)

node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Interactive Network Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    annotations=[dict(
                        text="Institution",
                        showarrow=False,
                        xref="paper", yref="paper"
                    )],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
                )

fig.show()
