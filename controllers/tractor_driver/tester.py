import torch
import torch.optim as optim
import open3d as o3d
import numpy as np


# Reuse the parametric arc function from our previous discussion
def create_parametric_arc(width, length, curvature, resolution=50):
    if abs(curvature) < 1e-6:
        # Simplified straight line for stability
        mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=0.01)
        return mesh
    
    radius_center = 1.0 / curvature
    theta_total = length * curvature
    r_inner = radius_center - (width / 2) * np.sign(radius_center)
    r_outer = radius_center + (width / 2) * np.sign(radius_center)
    
    vertices = []
    angles = np.linspace(0, theta_total, resolution)
    for alpha in angles:
        vertices.append([r_outer * np.sin(alpha), radius_center - r_outer * np.cos(alpha), 0])
        vertices.append([r_inner * np.sin(alpha), radius_center - r_inner * np.cos(alpha), 0])

    triangles = []
    for i in range(resolution - 1):
        v0, v1, v2, v3 = i*2, i*2+1, (i+1)*2, (i+1)*2+1
        triangles.append([v0, v2, v1])
        triangles.append([v1, v2, v3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    return mesh

# --- 1. GENERADOR DE MALLA PARA VISUALIZACIÓN ---
def create_mesh(params, color):
    w, l, k = params
    # Evitar k=0 exacto para la geometría
    k_adj = k if abs(k) > 0.001 else 0.001
    
    # Reutilizamos la lógica de creación de arco anterior
    radius_center = 1.0 / k_adj
    theta_total = l * k_adj
    r_inner = radius_center - (w / 2) * np.sign(radius_center)
    r_outer = radius_center + (w / 2) * np.sign(radius_center)
    
    res = 30
    vertices = []
    angles = np.linspace(0, theta_total, res)

    for a in angles:
        z_val = 0.01 if color[2] > 0.5 else 0.0 
        vertices.append([r_outer * np.sin(a), radius_center - r_outer * np.cos(a), z_val])
        vertices.append([r_inner * np.sin(a), radius_center - r_inner * np.cos(a), z_val])
    
    triangles = []
    for i in range(res - 1):
        v0, v1, v2, v3 = i*2, i*2+1, (i+1)*2, (i+1)*2+1
        triangles.append([v0, v2, v1]); triangles.append([v1, v2, v3])
        
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

# --- 2. MOTOR DE INFERENCIA ACTIVA (SDF) ---
class BrainSDF:
    def __init__(self):
        # Creencia inicial [ancho, largo, curvatura]
        self.mu = torch.tensor([0.3, 1.0, 0.01], requires_grad=True, dtype=torch.float32)
        self.optimizer = optim.Adam([self.mu], lr=0.01)

    def compute_loss(self, observation_points):
        y = torch.tensor(observation_points, dtype=torch.float32)
        w, l, k = self.mu[0], self.mu[1], self.mu[2]
        
        # SDF simplificada para el arco
        R_c = 1.0 / (k + 1e-6)
        dx = y[:, 0]
        dy = y[:, 1] - R_c
        dist_to_center = torch.sqrt(dx**2 + dy**2)
        
        # Error radial (Ancho)
        err_radial = torch.mean((dist_to_center - torch.abs(R_c)).abs() - w/2).pow(2)
        # Error de longitud (simplificado)
        loss = err_radial + 0.1 * ((w - 0.5)**2) # Prior: el robot prefiere ancho de 0.5
        return loss

    def update(self, points):
        self.optimizer.zero_grad()
        loss = self.compute_loss(points)
        loss.backward()
        self.optimizer.step()
        return self.mu.detach().numpy().copy()

# --- 3. SIMULACIÓN EN TIEMPO REAL ---

# Inicializar Visualizador
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Active Inference: Belief vs World", width=1024, height=768)

brain = BrainSDF()

# Parámetros del Mundo Real (van a cambiar)
world_params = [0.6, 2.0, 0.1] 
world_mesh = create_mesh(world_params, [0.2, 0.8, 0.2]) # Verde
belief_mesh = create_mesh(brain.mu.detach().numpy(), [0.2, 0.2, 0.8]) # Azul

vis.add_geometry(world_mesh)
vis.add_geometry(belief_mesh)

# Bucle de simulación
for i in range(10000):
    # A. El MUNDO cambia (La curva se vuelve más cerrada con el tiempo)
    world_params[2] += 0.0005 * np.sin(i / 50.0) 
    new_world = create_mesh(world_params, [0.2, 0.8, 0.2])
    world_mesh.vertices = new_world.vertices
    
    # B. PERCEPCIÓN: El robot "observa" puntos del mundo
    obs_points = np.asarray(new_world.sample_points_uniformly(200).points)
    
    # C. INFERENCIA: El cerebro actualiza su creencia (mu) para alinearse
    current_belief = brain.update(obs_points)
    
    # D. ACTUALIZAR MODELO INTERNO:
    new_belief = create_mesh(current_belief, [0.2, 0.2, 0.8])
    belief_mesh.vertices = new_belief.vertices
    
    # E. RENDERIZADO
    vis.update_geometry(world_mesh)
    vis.update_geometry(belief_mesh)
    vis.poll_events()
    vis.update_renderer()

    if i % 100 == 0:
        print(f"Mundo K: {world_params[2]:.3f} | Creencia K: {current_belief[2]:.3f}")

vis.destroy_window()