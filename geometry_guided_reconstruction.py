import json
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def midpoint(l=None, e1=None, e2=None):
    if l != None:
        x1, y1, x2, y2 = l
    else:
        x1, y1 = e1
        x2, y2 = e2
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def distanceBetweenPoints(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def indexOfClosestContourPoint(M, C_Y):
    dists = [distanceBetweenPoints(M, C) for C in C_Y]
    return dists.index(min(dists))

def makeLineFromLineSegment(l):
    x1, y1, x2, y2 = l
    a = y1 - y2
    b = x2 - x1
    c = x2 * y1 - x1 * y2
    return a, b, c

def pointOfIntersectionOfLines(O_i, O_j):
    l1 = makeLineFromLineSegment(O_i)
    l2 = makeLineFromLineSegment(O_j)

    d  = l1[0] * l2[1] - l1[1] * l2[0]
    dx = l1[2] * l2[1] - l1[1] * l2[2]
    dy = l1[0] * l2[2] - l1[2] * l2[0]
    if d == 0:
        # no point of intersection
        return None
    else:
        x = dx / d
        y = dy / d
        return [x, y]

def closestPoints(O_i, O_j):
    O_i = np.array(O_i)
    O_j = np.array(O_j)
    i1, i2 = O_i[0:2], O_i[2:4]
    j1, j2 = O_j[0:2], O_j[2:4]

    Vi = i2 - i1
    Vj = j2 - j1
    Vji = j1 - i1

    vjj = np.dot(Vj, Vj)
    vii = np.dot(Vi, Vi)
    vji = np.dot(Vj, Vi)
    vji_i = np.dot(Vji, Vi)
    vji_j = np.dot(Vji, Vj)
    d = vji * vji - vjj * vii

    if d == 0:
        s = 0
        t = (vii * s - vji_i) / vji
    else:
        s = (vji_j * vji - vjj * vji_i) / d
        t = (-vji_i * vji + vii * vji_j) / d

    s = max(0, min(s, 1))
    t = max(0, min(t, 1))

    return (i1 + s * Vi).tolist(), (j1 + t * Vj).tolist()

def closestEndpoints(O_i, O_j):
    i1, i2 = O_i[0:2], O_i[2:4]
    j1, j2 = O_j[0:2], O_j[2:4]
    pairs = [[i1, j1], [i1, j2], [i2, j1], [i2, j2]]
    dists = [distanceBetweenPoints(i, j) for i, j in pairs]
    return pairs[dists.index(min(dists))]

def closestPointOnLineSegment(M, O):
    x1, y1, x2, y2 = O
    x3, y3 = M
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    nx = ((x3 - x1) * dx + (y3 - y1) * dy) / d2
    nx = max(0, min(nx, 1))
    return [dx * nx + x1, dy * nx + y1]

def relativeOrderOfBuildingEdges(L_X, C_Y):
    O_X = [None] * len(C_Y)
    X = len(L_X)
    for x in range(X):
        L_x = L_X[x]
        M_x = midpoint(l=L_x)
        y_x = indexOfClosestContourPoint(M_x, C_Y)
        O_X[y_x] = L_x
    O_X = [O for O in O_X if O != None]
    return O_X

def buildingPolygonReconstruction(O_X):
    V_Z = []
    X = len(O_X)
    for i in range(X):
        j = (i + 1) % X
        O_i = O_X[i]
        O_j = O_X[j]
        I_ij = pointOfIntersectionOfLines(O_i, O_j)
        if I_ij != None:
            C_ij, C_ij_ = closestPoints(O_i, O_j)
            dist_ij = distanceBetweenPoints(I_ij, C_ij)
            dist_ij_ = distanceBetweenPoints(I_ij, C_ij_)
        if I_ij != None and min(dist_ij, dist_ij_) <= 10:
            V_Z.append(I_ij)
        else:
            E_ij, E_ij_ = closestEndpoints(O_i, O_j)
            M_ij = midpoint(e1=E_ij, e2=E_ij_)
            P_ij = closestPointOnLineSegment(M_ij, O_i)
            P_ij_ = closestPointOnLineSegment(M_ij, O_j)
            V_Z.append(P_ij)
            V_Z.append(P_ij_)
    return V_Z

def draw_figures(line_segments, contour_points, ordered_line_segments, polygon_vertex_points):
    img = mpimg.imread('813_im.png')
    cmap = plt.cm.get_cmap('gist_rainbow', len(polygon_vertex_points))
    figsize = 10
    linewidth = 2

    fig, axs = plt.subplots(2, 2, figsize=(figsize, figsize))
    for n, ax in enumerate(axs.flat):
        ax.imshow(img)
        ax.axis('off')
        if n == 0:
            ax.set_title('Line segments')
            for i, (x1, y1, x2, y2) in enumerate(line_segments):
                ax.add_line(lines.Line2D([x1, x2], [y1, y2], linewidth=linewidth, color='red'))
        elif n == 1:
            ax.set_title('Contour points')
            for i, (x, y) in enumerate(contour_points):
                ax.add_patch(patches.Circle((x, y), radius=linewidth/4, color='blue'))
        elif n == 2:
            ax.set_title('Ordered line segments')
            for i, (x1, y1, x2, y2) in enumerate(ordered_line_segments):
                ax.add_line(lines.Line2D([x1, x2], [y1, y2], linewidth=linewidth, color=cmap(i)))
        elif n == 3:
            ax.set_title('Polygon vertex points and line segments')
            for i, (x, y) in enumerate(polygon_vertex_points):
                ax.add_patch(patches.Circle((x, y), radius=linewidth, color=cmap(i), zorder=2))
            for i in range(len(polygon_vertex_points)):
                j = (i + 1) % len(polygon_vertex_points)
                x1, y1 = polygon_vertex_points[i]
                x2, y2 = polygon_vertex_points[j]
                ax.add_line(lines.Line2D([x1, x2], [y1, y2], linewidth=linewidth, color=cmap(i), zorder=1))
    fig.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)
    # list[x1, y1, x2, y2]
    line_segments = inputs['line_segments']
    # list[x, y]
    contour_points = inputs['contour_points']

    # list[x1, y1, x2, y2]
    ordered_line_segments = relativeOrderOfBuildingEdges(line_segments, contour_points)
    # list[x, y]
    polygon_vertex_points = buildingPolygonReconstruction(ordered_line_segments)

    with open('outputs.json', 'w') as f:
        json.dump({'polygon_vertex_points': polygon_vertex_points}, f)

    draw_figures(line_segments, contour_points, ordered_line_segments, polygon_vertex_points)
