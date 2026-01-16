import bpy
import numpy as np
import mathutils
import math

#blenderのスクリプト実行環境で外部モジュールをインポートする場合の例 folder_pathは自身のフォルダーパスに変更してください
# import sys
# folder_path = r"your_folder_path"
# sys.path.append(folder_path)
# import blender_p3p

# import importlib
# importlib.reload(blender_p3p)




#3次元座標点と対応する2次元画像点のセット
points_3d = [                
    (25.6,-674.3,-291.3),
    (827.2,-674.2,-295.0),
    (1718.0,-673.0,-289.1),
    (0,0,0),
    (1750.2,0.0,0.8),
    (0.4,5.6,850.0),
    (1750.9,7.1,853.2),
    (-1151.3,1436.2,1303.2),
    (-180.1,1467.8,1292.1),
]

points_2d = [
    (2100, 3962),
    (4135, 4068),
    (6322, 4160),
    (2343, 3120),
    (5960, 3316),
    (2450, 1332),
    (6062, 1575),
    (1003, 873),
    (2530, 1007),
]

points_len = len(points_3d)

np.set_printoptions(precision=5, suppress=True)
if "unregister" in locals():  
    try:
        unregister()
    except:
        pass

my_tag = "p3p_addon"

def clear_spawned_objects():
    for obj in bpy.data.objects:
        if my_tag in obj.keys():
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge()


class Calculate(bpy.types.Operator):
    bl_idname = "dg3.camera"
    bl_label = "Calculate P3P"

    def execute(self, context):
        selected_indexes = []
        for i in range(points_len):
            if getattr(context.scene, f"p3p_select_{i}"):
                selected_indexes.append(i)
                
        if len(selected_indexes) == 3:
            clear_spawned_objects()
            solution_num = main(selected_indexes)
            self.report({'WARNING'}, f"Found {solution_num} solutions.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Select exactly 3 points for P3P")
            return {'CANCELLED'}


class P3P(bpy.types.Menu):
    bl_label = "P3P"
    bl_idname = "P3P_MT_top_menu"

    def draw(self, context):
        layout = self.layout
        layout.operator(Calculate.bl_idname)

        layout.label(text="Select Points:")

        for i in range(points_len):
            layout.prop(context.scene, f"p3p_select_{i}")


def menu_draw(self, context):
    self.layout.menu(P3P.bl_idname)

classes = (
    P3P,
    Calculate,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    for i in range(points_len):
        setattr(
            bpy.types.Scene,
            f"p3p_select_{i}",
            bpy.props.BoolProperty(name=f"Point {i}")
        )

    bpy.types.TOPBAR_MT_editor_menus.append(menu_draw)


def unregister():
    bpy.types.TOPBAR_MT_editor_menus.remove(menu_draw)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    for i in range(points_len):
        delattr(bpy.types.Scene, f"p3p_select_{i}")


register()
print("p3p addon registered")


class Solution:
    def __init__(self, R, C):
        self.R = R
        self.C = C 

def rq(A):
    A = np.asarray(A)
    inv = np.array([[0,0,1],[0,1,0],[1,0,0]])
    A_ = inv @ A @ inv
    Q_, R_ = np.linalg.qr(A_.T)
    R = inv @ R_.T @ inv
    Q = inv @ Q_.T @ inv

    if np.linalg.det(Q) < 0:
        Q *= -1
        R *= -1

    return R, Q

def calibrate_camera(points_3d, points_2d):
    n = len(points_3d)
    A = []
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # 零空間

    for (X, Y, Z), (u, v) in zip(points_3d, points_2d):
        vec = np.array([X, Y, Z, 1])
        proj = P @ vec
        proj /= proj[2]  # perspective除算
        print(f"3D: {(X,Y,Z)} -> projected {(proj[0], proj[1])}, target {(u,v)}")

    return P

def decompose_projection_matrix(P):
    H = P[:, :3]
    K, R = rq(H)
    print(f'Decomposed R:\n{R}')
    print(f'Decomposed K:\n{K}')

    scale = K[2, 2]
    K /= scale  
    P /= scale  

    t = P[:, 3]
    t = -np.linalg.inv(H) @ (t)

    return K, R, t

def p3p_solver(points_3d, points_2d, K):
    m = []
    for i, (u, v) in enumerate(points_2d): #正規化座標
        p = np.array([u, v, 1.0])
        m_tild = np.linalg.inv(K) @ p
        m.append(m_tild / np.linalg.norm(m_tild))

    AB = np.linalg.norm(np.array(points_3d[0]) - np.array(points_3d[1]))
    AC = np.linalg.norm(np.array(points_3d[0]) - np.array(points_3d[2]))
    BC = np.linalg.norm(np.array(points_3d[1]) - np.array(points_3d[2]))
    a = (AB**2) / (BC**2)
    b = (AC**2) / (BC**2)

    m12 = m[0].T @ m[1]
    m13 = m[0].T @ m[2]
    m23 = m[1].T @ m[2]

    #四次方程式の係数
    coeffs = [a**2 - 2*a*b - 2*a + b**2 - 4*b*m12**2 + 2*b + 1, -4*a**2*m23 + 8*a*b*m23 + 4*a*m12*m13 + 4*a*m23 - 4*b**2*m23 + 8*b*m12**2*m23 + 4*b*m12*m13 - 4*b*m23 - 4*m12*m13, 4*a**2*m23**2 + 2*a**2 - 8*a*b*m23**2 - 4*a*b - 8*a*m12*m13*m23 - 4*a*m13**2 + 4*b**2*m23**2 + 2*b**2 - 4*b*m12**2 - 8*b*m12*m13*m23 + 4*m12**2 + 4*m13**2 - 2, -4*a**2*m23 + 8*a*b*m23 + 4*a*m12*m13 + 8*a*m13**2*m23 - 4*a*m23 - 4*b**2*m23 + 4*b*m12*m13 + 4*b*m23 - 4*m12*m13, a**2 - 2*a*b - 4*a*m13**2 + 2*a + b**2 - 2*b + 1]

    roots_y = np.roots(coeffs)

    for r in roots_y:
        print(r)

    solution = []
    print('---')   
    for y in roots_y:

        x = (2*a*m23*y - a*y**2 - a - 2*b*m23*y + b*y**2 + b + y**2 - 1)/(2*(m12*y - m13))

        d3 = AC / np.sqrt(x**2 - 2*m13*x + 1)
        d1 = x * d3
        d2 = y * d3

        EPSILON = 1e-8

        if d1.real < EPSILON or d2.real < EPSILON or d3.real < EPSILON:
            continue

        if abs(d1.imag) > EPSILON or abs(d2.imag) > EPSILON or abs(d3.imag) > EPSILON:
            continue

        rot_left = []
        rot_left.append(d1 * m[0] - d2 * m[1])
        rot_left.append(d3 * m[2] - d1 * m[0])
        rot_left.append(np.cross(rot_left[0], rot_left[1]))

        rot_right = []
        rot_right.append(np.array(points_3d[0]) - np.array(points_3d[1]))
        rot_right.append(np.array(points_3d[2]) - np.array(points_3d[0]))
        rot_right.append(np.cross(rot_right[0], rot_right[1]))

        R = np.column_stack(rot_left) @ np.linalg.inv(np.column_stack(rot_right))

        t = d1 * m[0] - R @ np.array(points_3d[0])
        C = -R.T @ t
        print(f'Camera center C: {C}')

        Rt = np.column_stack((R, t))
        P_est = K @ Rt
        print(f'Reprojection matrix: {P_est}')
        for (X, Y, Z), (u, v) in zip(points_3d, points_2d):
            vec = np.array([X, Y, Z, 1])
            proj = P_est @ vec
            proj /= proj[2] 
            print(f"3D: {(X,Y,Z)} -> projected {(proj[0], proj[1])}, target {(u,v)}")
        print('---')    
        solution.append(Solution(R.real, C.real))
          
    return solution


def spawn_camera(fov_x):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens_unit = 'FOV'
    cam.data.angle_x = fov_x
    cam.data.display_size = 10
    cam[my_tag] = True
    return cam

def spawn_spheres(points_3d, p3p_indexes):
    for i, (x, y, z) in enumerate(points_3d):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x / 100, y / 100,  z / 100))
        sphere = bpy.context.active_object
        sphere.name = f'Point_{p3p_indexes[i]}'
        sphere[my_tag] = True

def main(p3p_indexes):
    #画像サイズ
    width, height = 8256, 5504
    cx, cy = width / 2, height / 2

    for i, (u, v) in enumerate(points_2d):
        points_2d[i] = (u - cx, (height - v) - cy)

    P = calibrate_camera(points_3d, points_2d)
    print(f'P:\n{P}')

    K, R, C = decompose_projection_matrix(P)
    print(f'K:\n{K}')
    print(f'R:\n{R}')
    print(f'C:\n{C}')

    p3p_points_3d = [points_3d[i] for i in p3p_indexes]
    p3p_points_2d = [points_2d[i] for i in p3p_indexes]

    solutions = p3p_solver(p3p_points_3d, p3p_points_2d, K)

    fov_x = 2 * np.arctan(width / (2 * K[0,0]))

    spawn_spheres(p3p_points_3d, p3p_indexes)

    for s in solutions:
        cam = spawn_camera(fov_x)
        cam.location = s.C / 100
        cam.rotation_mode = 'QUATERNION'
        rot = mathutils.Matrix(s.R.T.tolist()).to_quaternion()
        cam.rotation_quaternion =  rot @ mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(180))  

    return len(solutions)

