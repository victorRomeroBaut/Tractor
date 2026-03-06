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


class ActiveInferenceController:
    def __init__(self):
        # --- PREFERENCIA (Lo que el robot desea ver) ---
        # Un arco recto, centrado y estable
        self.preferred_mu = torch.tensor([0.51, 2.8, 0.0], dtype=torch.float32)
        
        # --- ACCIÓN (Lo que optimizamos: Pose Relativa) ---
        # [x, y, theta]
        self.action = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, dtype=torch.float32)
        # Aumentamos el Learning Rate para que el movimiento sea más agresivo
        self.optimizer = optim.Adam([self.action], lr=0.005)

    def get_loss(self, world_points):
        # 1. Aplicar la ACCIÓN a los puntos del mundo
        x, y, theta = self.action[0], self.action[1], self.action[2]
        
        # Matriz de rotación 2D
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Transformación: Primero rotamos, luego trasladamos (Sistema coordenadas Robot)
        # x' = x*cos + y*sin - tx
        # y' = -x*sin + y*cos - ty
        p_x = world_points[:, 0] * cos_t + world_points[:, 1] * sin_t - x
        p_y = -world_points[:, 0] * sin_t + world_points[:, 1] * cos_t - y
        
        # 2. SDF del Arco Deseado (Recto)
        w_p, l_p, k_p = self.preferred_mu
        
        # Error lateral (distancia al eje Y central)
        err_x = torch.abs(p_x) - (w_p / 2.0)
        err_x = torch.clamp(err_x, min=0)
        
        # Error longitudinal (distancia al segmento de longitud L)
        err_y = torch.abs(p_y - l_p/2.0) - (l_p / 2.0)
        err_y = torch.clamp(err_y, min=0)
        
        # Error total (Distancia Euclidiana al sólido)
        # Usamos una suma simple para que el gradiente sea más lineal y no se desvanezca
        dist_sq = err_x + err_y 
        
        return torch.mean(dist_sq)

    def step(self, points):
        self.optimizer.zero_grad()
        loss = self.get_loss(torch.tensor(points, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
        return self.action.detach().numpy()



# 2. El Controlador (Active Inference)
controller = ActiveInferenceController()

# 3. Visualización
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Active Inference: Action Control", width=800, height=600)

# Mundo: Una fila desplazada 1 metro a la derecha y rotada
world_params = [0.5, 3.0, 0.0] 
world_mesh = create_mesh(world_params, [0.2, 0.8, 0.2])
world_mesh.translate([1.0, 0.5, 0]) # Desplazamiento inicial real
world_mesh.rotate(world_mesh.get_rotation_matrix_from_xyz((0, 0, 0.4)), center=(0,0,0))

# Creencia/Deseo: Arco fijo en el centro (Azul)
controller = ActiveInferenceController()
target_mesh = create_mesh(controller.preferred_mu.tolist(), [0.1, 0.1, 0.9])

vis.add_geometry(world_mesh)
vis.add_geometry(target_mesh)

for i in range(15000):
    # 1. El robot "mira" el mundo
    obs_points = np.asarray(world_mesh.sample_points_uniformly(300).points)[:, :2]
    
    # 2. Inferencia Activa: ¿Qué acción reduce la sorpresa?
    # Obtenemos la pose que el robot DEBERÍA tener para que el mundo encaje
    pose = controller.step(obs_points)
    
    # 3. Aplicar movimiento al mundo (Simulando que el robot se mueve hacia la pose)
    # En un robot real, aquí enviarías comandos de velocidad. 
    # Aquí movemos la malla del mundo en sentido inverso al movimiento del robot.
    tx, ty, tr = pose * 0.05 # Movemos un pequeño paso hacia la solución
    
    world_mesh.translate([-tx, -ty, 0])
    R = world_mesh.get_rotation_matrix_from_xyz((0, 0, -tr))
    world_mesh.rotate(R, center=(0,0,0))
    
    # Resetear la acción interna para que la optimización sea incremental
    with torch.no_grad():
        controller.action.zero_()

    if i % 10 == 0:
        vis.update_geometry(world_mesh)
        vis.poll_events()
        vis.update_renderer()

vis.destroy_window()