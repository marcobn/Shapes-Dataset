"""
Generates images of different shapes in grayscale similar to MNIST.
    Circles
    Ellipses
    Parabolas
    Hyperbolas
    Triangles
    (non solid NGons) # Not via command line

The images are are written with their classification
gzip-ed to a .csv file as a flat vector.

By default the resolution is 28x28 pixels that need to be reshaped again.

The shapes are randomly rotated, translate and scaled.
For circles, triangles and ellipses a minimum area is set.
For ellipses also a minimum relative differences between the radii 
to not be randomly equal to a circle is implemented.

---

Command Line usage
required: Argument 1 the word 'run' to enable it.
optional: Argument 2 choose subset size, default 2000
optional: Argument 3 Outputfile name

for example this generates 5 * 1000 images:
python shapes_generation.py run 1000

---

Functional use:

make<Shape>Sample 
    plots a random sample of that shape.
    
<Shape>Generator 
    is present for all shape that yields x,y data for one image
    
create<Shape>ImageDataset 
    creates .png files of the images
    
create<Shape>NumericImageDataset
    creates one .csv files like described above

create<Shape>NumericalDataset
    creates a non image data set of a few X/Y pairs
    Not available for triangles and NGons

createGrayscaleImageDataset(List[generators], 
                            names : List[str], 
                            subset_size : int =DEFAULT_SUBSET_SIZE, 
                            resolution : Tuple[int, int] = (28, 28) )
    creates a full dataset of different shapes from generator objects

-----

Parts are based on or taken from:
https://github.com/cctech-labs/ml-2dshapes/blob/master/Conic_Shapes_Generator.ipynb
Their generation creates a numerical dataset of XY values.

-----

WIP:
Letter generator untested.

TODOS:
- Store as numpy .npz files.
- Clean up the file and more pythonic naming.
- Minimal storage option: Only save latent data, for example 3 points
  for a triangle and create the images from that.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import argparse


# Default subset size (legacy CLI previously overrode this dynamically)
DEFAULT_SUBSET_SIZE = 2000

#

DEFAULT_RESOLUTION = (28, 28)  # kept as a constant, but CLI --resolution is now the runtime default

__IMG_COUNTER = 0 # Needed for tracking, depracted

Random_Generator = np.random.default_rng()

def set_seed(seed: int | None):
    """Set global RNG seed for deterministic dataset generation."""
    global Random_Generator
    if seed is not None:
        Random_Generator = np.random.default_rng(seed)

def ax_plot_digit(ax, digit, shape=DEFAULT_RESOLUTION, colormap=plt.cm.gray, aspect=None):
    ax.imshow(digit.reshape(shape), cmap=colormap, aspect=aspect)#extent=[-4,4,-1,1], )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_digit(data):
    fig, ax = plt.subplots()
    ax_plot_digit(ax, data)
    plt.show()
       

def viewer(x_data, y_data, title, save=False):
    """
    Plots with standard matplotlib for demonstration, saves as .svg if wanted.
    """
    fig = plt.figure(figsize=[10,10])
    plt.plot(x_data,y_data,'b--')
    plt.xlabel('X-axis',fontsize=14)
    plt.ylabel('Y-axis',fontsize=14)
    plt.ylim(-20, 20)
    plt.xlim(-20, 20)
    plt.axhline(y=0, color ="k")
    plt.axvline(x=0, color ="k")
    plt.grid(True)
    if save:
        saveFile = title + '.svg'
        plt.savefig(saveFile)
    plt.show()
    return fig

# =============================================================================

def __plot_normal(ax, x_data, y_data, color='black', linewidth=2):
    ax.plot(x_data, y_data, color=color, linewidth=linewidth, antialiased=True)
    
def __plot_hyperbola(ax, x_data, y_data, color='black', linewidth=2):
    # Need to sepearate the plots else the jump lines will be plotted too
    l = len(x_data)
    # Arc 1
    ax.plot(x_data[:l//4], y_data[:l//4], color=color, linewidth=linewidth, antialiased=True)
    ax.plot(x_data[l*3 // 4:], y_data[l*3 // 4:], color=color, linewidth=linewidth, antialiased=True)
    # Arc 2
    ax.plot(x_data[l//4 : l*3 // 4], y_data[l//4 : l*3 // 4], color=color, linewidth=linewidth, antialiased=True)


def __plot_letter(ax, x_data, y_data, plot_letter, color='black'):
    ax.text(x_data, y_data, plot_letter, color=color, fontsize='xx-large')
    

def saving_plotter(x_data, y_data, folder="Dataset/", resolution=DEFAULT_RESOLUTION, plot_hyperbola=False):
    """
    Depracted
    Saves image as .png
    """
    fig = plt.figure(figsize=(resolution[0] / 10, resolution[0] / 10), edgecolor=None, frameon=False, dpi=10)
    fig.patch.set_visible(False)
    plt.axis('off')
    plt.box(False)
    fig.tight_layout(pad=0)
    if plot_hyperbola:
        __plot_hyperbola(x_data, y_data)
    else:
        __plot_normal(x_data, y_data) 
    plt.ylim(-20, 20)
    plt.xlim(-20, 20)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    global __IMG_COUNTER
    plt.savefig(r"images\img"+str(__IMG_COUNTER)+".png", dpi=10)
    plt.show()
    plt.close()
    __IMG_COUNTER += 1
    return fig

# =============================================================================
# Plot to grayscale matrix.


def numeric_save_plotter(x_data, y_data, resolution=DEFAULT_RESOLUTION, show=False, plot_hyperbola=False, plot_letter=""):
    """Render the provided (x,y) data to a grayscale numpy array with shape `resolution`.

    Previous implementation created a very large high-DPI canvas (e.g. 2800x2800 px for a 28x28 target)
    and then attempted to forcibly reshape the raw buffer to the small resolution which raised a
    ValueError. We now create a canvas whose pixel dimensions match the requested resolution so the
    buffer can be reshaped safely. As a fallback, if for any reason the canvas size differs, we perform
    a simple nearest-neighbour down/upsampling via slicing / numpy repeat (without adding new deps).
    """
    target_h, target_w = resolution  # (rows, cols)

    # Choose a dpi and figsize combination that yields exactly the target pixel resolution.
    # pixel_width = figsize[0] * dpi, pixel_height = figsize[1] * dpi
    # We keep dpi moderate (e.g. 100) to avoid backend / font issues and set figsize accordingly.
    dpi = 100
    figsize = (target_w / dpi, target_h / dpi)
    fig = plt.figure(figsize=figsize, edgecolor=None, frameon=False, dpi=dpi)
    # Transparent figure and axes to ensure background stays 0 in RGB
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed axes
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))
    plt.margins(0)
    line_color = 'white'
    # Scale line width for resolution; keep thin lines for 28x28
    lw = max(1.0, (min(target_h, target_w) / 28.0) * 2.0)
    if plot_hyperbola:
        __plot_hyperbola(ax, x_data, y_data, color=line_color, linewidth=lw)
    elif not plot_letter:
        __plot_normal(ax, x_data, y_data, color=line_color, linewidth=lw)
    else:
        __plot_letter(ax, x_data, y_data, plot_letter, color=line_color)
    ax.set_ylim(-20, 20)
    ax.set_xlim(-20, 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # tight_layout can interfere with full-bleed axes; not needed
    fig.canvas.draw()

    # Obtain buffer & actual canvas dimensions
    buf, (canvas_w, canvas_h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype='uint8').reshape(canvas_h, canvas_w, 4)
    # Use alpha channel so background (fully transparent) is 0 and strokes are >0
    img_mat = img[:, :, 3]

    # If the canvas size does not match target resolution, resize (nearest neighbour) without extra deps
    if (canvas_h, canvas_w) != (target_h, target_w):
        # Simple scaling using numpy indexing / repetition
        scale_y = target_h / canvas_h
        scale_x = target_w / canvas_w
        if scale_y < 1 or scale_x < 1:
            # Downscale by selecting nearest indices
            y_idx = (np.linspace(0, canvas_h - 1, target_h)).astype(int)
            x_idx = (np.linspace(0, canvas_w - 1, target_w)).astype(int)
            img_mat = img_mat[np.ix_(y_idx, x_idx)]
        else:
            # Upscale by repeating
            img_mat = np.repeat(np.repeat(img_mat, int(np.ceil(scale_y)), axis=0), int(np.ceil(scale_x)), axis=1)[:target_h, :target_w]

    if show:
        plt.imshow(img_mat, cmap='gray')
        plt.axis('off')
        plt.show()
    plt.close(fig)
    return img_mat

# =============================================================================

# General functions

def rotateCoordinates(x_data, y_data, rot_angle):
    x_ = x_data*math.cos(rot_angle) - y_data*math.sin(rot_angle)
    y_ = x_data*math.sin(rot_angle) + y_data*math.cos(rot_angle)
    return x_,y_

def get_n_samples(x_data, y_data, n):
    indexes = np.round(np.linspace(0, 99, n)).astype('int')
    return x_data[indexes], y_data[indexes]

#returns a  single random index from an array



def get_random_index(array_size):
        return Random_Generator.integers(array_size)
        #return 5
    
def build_dataset(x_,y_, shape):
        row = {}
        data = []
        for i in range(len(x_)):
            row['x' + str(i+1)] = x_[i]
            row['y' + str(i+1)] = y_[i]
        row['shape'] = shape  
        data.append(row)  
        return data
  
# =============================================================================

    
def createParabola(focal_length, center, rotation):
    t = np.linspace(-math.pi*2, math.pi*2, 100)
    x_parabola = focal_length * t**2
    y_parabola = 2 * focal_length * t
    if rotation is not None:
        x_parabola, y_parabola = rotateCoordinates(x_parabola, y_parabola, rotation) 
    x_parabola = x_parabola + center[0]
    y_parabola = y_parabola + center[1]
    return x_parabola, y_parabola

def makeParabolaSample():
    angle = [0, math.pi/4, math.pi*2/3, math.pi*4/3]
    j=0
    for i in angle:
        j=j+1
        x_parabola, y_parabola = createParabola(focal_length=1.8, center=[-3+j,-4+j], rotation=i)
        temp = 'Parabola '+ str(j)
        viewer(x_parabola, y_parabola, temp)

# =============================================================================
    
def createCircle(radius, center):
    theta = np.linspace(0, 2*math.pi,100)
    x_circle = radius * np.cos(theta) + center[0]
    y_circle = radius * np.sin(theta) + center[1]
    return x_circle, y_circle

def makeCircleSample():
    center = [[0,0],[-1,-2],[2,-1.5],[-1.8,1.2]]
    for i in center:
        x_circle, y_circle = createCircle(center=i, radius=15)
        viewer(x_circle,y_circle, 'Circle')

# =============================================================================    
    
def createEllipse(major_axis, minor_axis, center, rotation):
    theta = np.linspace(0, 2*math.pi, 100)
    x_ellipse = major_axis * np.cos(theta) 
    y_ellipse = minor_axis * np.sin(theta) 
    if rotation is not None:
        x_ellipse, y_ellipse = rotateCoordinates(x_ellipse,y_ellipse, rotation)
    x_ellipse = x_ellipse + center[0]
    y_ellipse = y_ellipse + center[1]
    return x_ellipse, y_ellipse

def makeEllipseSample():
    angle = [0, math.pi/4, math.pi*2/3, math.pi*4/3]
    j=0
    for i in angle:
        j=j+1
        x_ellipse, y_ellipse = createEllipse(major_axis=16, minor_axis=8, center=[-1+j, -1.5+j], rotation=i)
        temp = 'Ellipse' +' '+ str(j)
        viewer(x_ellipse,y_ellipse, temp)
    
# ============================================================================= 

def createHyperbola(major_axis, conjugate_axis, center, rotation):
    theta = np.linspace(0, 2*math.pi, 100)
    x_hyperbola = major_axis * 1/np.cos(theta) + center[0]
    y_hyperbola = conjugate_axis * np.tan(theta) + center[1]
    if rotation is not None:
        x_hyperbola, y_hyperbola = rotateCoordinates(x_hyperbola, y_hyperbola, rotation)
    x_hyperbola = x_hyperbola #+ center[0]
    y_hyperbola = y_hyperbola #+ center[1]
    return x_hyperbola, y_hyperbola

def makeHyperbolaSample():
    j = 0
    angle = [0, math.pi/4, math.pi*2/3, math.pi*4/3]
    for i in angle:
        j=j+1;
        x_hyperbola, y_hyperbola = createHyperbola(major_axis=5, conjugate_axis=3, center=[-2+j,0+j],rotation=i)
        temp = 'Hyperbola' +' '+ str(j)
        viewer(x_hyperbola,y_hyperbola, temp)


# =============================================================================


def createTriangle(s1, s2, ang, center, rotation):
    # Building triangles by SWS
    pass


# =============================================================================
# Generate Dataset
# =============================================================================


def AdvTriangleGenerator(amount=DEFAULT_SUBSET_SIZE):
    """
    NOT USED
    """
    ## angle in [15° to 150°]
    angles = np.linspace(math.pi / 6, 5/3 * math.pi , 100)
    ## lengths in [1 to 10]
    side_lengths = np.linspace(1, 10, 100)
    
    center_x_arr = np.linspace(-8, 8, 100)
    center_y_arr = np.linspace(-8, 8, 100)
    
    
    rotation_array = np.linspace(2*math.pi, 100)
    
    print("Generating Triangles")
    for i in range(amount):
        s1 = side_lengths[get_random_index(len(side_lengths))]
        s2 = side_lengths[get_random_index(len(side_lengths))]
        a = angles[get_random_index(len(angles))]
        
        center_x = center_x_arr[get_random_index(len(center_x_arr))]
        center_y = center_y_arr[get_random_index(len(center_y_arr))]
        rotation = rotation_array[get_random_index(len(rotation_array))]
        
        x, y = createTriangle(s1, s2, a, center=(center_x, center_y), rotation=rotation)
        yield x, y

# =============================================================================

def calc_triangle_area(points):
    """
    Expected input shape is (3, d). d The dimension.
    """
    Vs = np.subtract(points, points[0])
    cross = np.cross(Vs[1], Vs[2])
    return 0.5 * np.linalg.norm(cross)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def get_angles(points):
    a1 = angle_between(points[0], points[1])
    a2 = angle_between(points[0], points[2])
    a3 = 180 - a1 - a2
    return np.abs((a1, a2, a3))

def SimpleNGonGenerator(n, amount=DEFAULT_SUBSET_SIZE, min_area=15):
    """
    Works good for traingles, but points are not ordered so quads and higher
    are not corret.
    """
    locations = np.linspace(-15, 15, 100)
    # return 3 samples each
    
    print("Generating NGons, n =", n)
    
    for i in range(amount):
        area = 0
        angles = np.array([0, 0, 0])
        # Optional get triangles with a minimal area and angles to prevent lines.
        while (area < min_area or (not min_area)) or not (angles > 15).all():
            # Straight line ngons possible, without further filtering. TODO
            x = [locations[get_random_index(len(locations))] for _ in range(n)]
            y = [locations[get_random_index(len(locations))] for _ in range(n)]
            # NOTE, todo: Optimized for traingles!
            points = np.vstack((x,y)).T
            area = calc_triangle_area(points)
            angles = get_angles(points)

        # repeat first element for plotting last connection
        x.append( x[0] )
        y.append( y[0] )
        yield x, y

def NGonSample(n,  size):
    for x, y in SimpleNGonGenerator(n, size):
        viewer(x, y, str(n) + '-gon')    

# =============================================================================

def TriangleGenerator(amount=DEFAULT_SUBSET_SIZE, min_area=40):
    return SimpleNGonGenerator(n=3, amount=amount, min_area=min_area)

def TriangleSample(size):
    for x, y in TriangleGenerator(size):
        viewer(x, y, 'triangle')
      
def createTriangleNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION, show=False, min_area=40):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')

    for i, (x, y) in enumerate(TriangleGenerator(amount, min_area)):
        data[i] = numeric_save_plotter(x, y, resolution, show).flatten()
    
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])
    df.insert(0, 'shape', 'triangle')
    return df


# =============================================================================

def ParabolaGenerator(amount=DEFAULT_SUBSET_SIZE):
    # Parabola
    focal_length_array = np.linspace(1, 10, 100)
    center_x_arr = np.linspace(-8, 8, 100)
    center_y_arr = np.linspace(-8, 8, 100)
    rotation_array = np.linspace(2*math.pi, 100)
    
    print("Generating Parabolas")
    for i in range(amount):
        focal_length = focal_length_array[get_random_index(len(focal_length_array))]
        center_x = center_x_arr[get_random_index(len(center_x_arr))]
        center_y = center_y_arr[get_random_index(len(center_y_arr))]
        rotation = rotation_array[get_random_index(len(rotation_array))]
        x, y = createParabola(focal_length= focal_length, center=(center_x, center_y), rotation=rotation)
        yield x, y

def createParabolaNumericalDataset(sample_count=6):
    # Parabola
    gen = ParabolaGenerator(amount=DEFAULT_SUBSET_SIZE)
    # Make dataframe from first result
    x, y = gen.__next__()
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'parabola')
    parabola_dataset = pd.DataFrame.from_dict(data)
    
    for i, (x, y) in enumerate(gen):
        x_, y_ = get_n_samples(x, y, sample_count)
        data = build_dataset(x_, y_, 'parabola')
        parabola_dataset.loc[i, :] = data.values()
    return parabola_dataset

def createParabolaImageDataset():
    for x, y in ParabolaGenerator(amount=DEFAULT_SUBSET_SIZE):
        saving_plotter(x, y)
        
def createParabolaNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')

    for i, (x, y) in enumerate(ParabolaGenerator(amount)):
        data[i] = numeric_save_plotter(x, y, resolution).flatten()
    
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])
    df.insert(0, 'shape', 'parabola')
    return df
    

# =============================================================================

# Ellipse
def EllipseGenerator(amount=DEFAULT_SUBSET_SIZE):
    major_axis_array = np.linspace(1.5, 15,100)
    minor_axis_array = np.linspace(1.5, 15,100)
    center_x_arr = np.linspace(-10, 10, 100)
    center_y_arr = np.linspace(-10, 10, 100)
    rotation_array = np.linspace(2*math.pi, 100)
    
    print("Generating Ellipses")
    for i in range(amount):
        major_axis = major_axis_array[get_random_index(len(major_axis_array))]
        minor_axis = minor_axis_array[get_random_index(len(minor_axis_array))]
        while math.isclose(major_axis, minor_axis, rel_tol=0.15, abs_tol=0.3):
            # rerandom if to close to circle
            minor_axis = minor_axis_array[get_random_index(len(minor_axis_array))]
        center_x = center_x_arr[get_random_index(len(center_x_arr))]
        center_y = center_y_arr[get_random_index(len(center_y_arr))]
        rotation = rotation_array[get_random_index(len(rotation_array))]
        x,y = createEllipse(major_axis=major_axis, minor_axis=minor_axis, center= [center_x,center_y], rotation=rotation)
        yield x, y


def createEllipseNumericalDataset(sample_count=6):
    gen = EllipseGenerator(amount=DEFAULT_SUBSET_SIZE)
    # Make dataframe from first result
    x, y = gen.__next__()
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'ellipse')
    
    ellipse_dataset = pd.DataFrame.from_dict(data) 
    for i, (x, y) in enumerate(gen):
        x_,y_ = get_n_samples(x, y, sample_count)
        data = build_dataset(x_, y_, 'ellipse')[0]
        ellipse_dataset.loc[i, :] = data.values()
    return ellipse_dataset


def createEllipseNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION, show=False):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')

    for i, (x, y) in enumerate(EllipseGenerator(amount)):
        data[i] = numeric_save_plotter(x, y, resolution, show).flatten()
    
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])
    df.insert(0, 'shape', 'ellipse')
    return df
    

# =============================================================================

# Hyperbola
def HyperbolaGenerator(amount=DEFAULT_SUBSET_SIZE):
    major_axis_array = np.linspace(1, 15, 100)
    conjugate_axis_array = np.linspace(1, 15,100)
    center_x_arr = np.linspace(-7, 7, 100)
    center_y_arr = np.linspace(-7, 7, 100)
    rotation_array = np.linspace(2*math.pi, 100)
    
    print("Generating Hyperbolas")
    for i in range(amount):
        major_axis = major_axis_array[get_random_index(len(major_axis_array))]
        conjugate_axis = conjugate_axis_array[get_random_index(len(conjugate_axis_array))]
        center_x = center_x_arr[get_random_index(len(center_x_arr))]
        center_y = center_y_arr[get_random_index(len(center_y_arr))]
        rotation = rotation_array[get_random_index(len(rotation_array))]
        x, y = createHyperbola(major_axis=major_axis, conjugate_axis=conjugate_axis, center= [center_x,center_y], rotation=rotation)
        yield (x, y)


def createHyperbolaNumericalDataset(sample_count=6):
    gen = HyperbolaGenerator(amount=DEFAULT_SUBSET_SIZE)
    # Make dataframe from first result
    x, y = gen.__next__()
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'hyperbola')
    
    hyperbola_dataset = pd.DataFrame.from_dict(data)
    for i, (x, y) in enumerate(gen):
        x_,y_ = get_n_samples(x, y, sample_count)
        data = build_dataset(x_, y_, 'hyperbola')[0]
        hyperbola_dataset.loc[i, :] = data.values()
    return hyperbola_dataset

def createHyperbolaImageDataset(amount=DEFAULT_SUBSET_SIZE):
    for x, y in HyperbolaGenerator(amount):
        saving_plotter(x, y)
    
def createHyperbolaNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION, show=False):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')
    
    for i, (x, y) in enumerate(HyperbolaGenerator(amount)):
        data[i] = numeric_save_plotter(x, y, resolution, show=show, plot_hyperbola=True).flatten()

    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])
    df.insert(0, 'shape', 'hyperbola')
    return df
    
    
# =============================================================================

# Circle
def CircleGenerator(amount=DEFAULT_SUBSET_SIZE):
    radius_array = np.linspace(2, 15, 100)
    center_x_arr = np.linspace(-10, 10, 100)
    center_y_arr = np.linspace(-10, 10, 100)
        
    print("Generating Circles")
    for i in range(amount):
        radius = radius_array[get_random_index(len(radius_array))]
        center_x = center_x_arr[get_random_index(len(center_x_arr))]
        center_y = center_y_arr[get_random_index(len(center_y_arr))]
        x, y = createCircle(radius = radius, center= [center_x,center_y])
        yield x, y

def createCircleNumericalDataset(sample_count=6):
    gen = CircleGenerator(amount=DEFAULT_SUBSET_SIZE)
    # Make dataframe from first result
    x, y = gen.__next__()
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'circle')
    circle_dataset = pd.DataFrame.from_dict(data)
    
    for i, (x, y) in enumerate(gen):
        x_,y_ = get_n_samples(x, y, sample_count)
        data = build_dataset(x_, y_, 'circle')
        circle_dataset.loc[i, :] = data.values()
    return circle_dataset

def createCircleImageDataset(amount):
    for x, y in CircleGenerator(amount):
        saving_plotter(x, y)
        

def createCircleNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')

    for i, (x, y) in enumerate(CircleGenerator(amount)):
        data[i] = numeric_save_plotter(x, y, resolution).flatten()
 
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])        
    df.insert(0, 'shape', 'circle')
    return df
        

# =============================================================================
# Font
# =============================================================================

# Letter
def LetterGenerator(amount, type="upper"):
    """
    Yields only a string containing a single character.
    """
    center_x_arr = np.linspace(-10, 10, 100)
    center_y_arr = np.linspace(-10, 10, 100)
    
    if type == 'upper':
        alphabet = list(map(chr, range(ord('A'), ord('Z')+1)))
    for _ in range(amount):
        center_x = center_x_arr[get_random_index(100)] #NOTE Hardcoded sizes!
        center_y = center_y_arr[get_random_index(100)]
        yield center_x, center_y, alphabet[Random_Generator.integers(26)]
    

def createLetterNumericImageDataset(amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION, show=False):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')


    for i, (x, y, letter) in enumerate(LetterGenerator(amount)):
        print(letter, type(letter))
        data[i] = numeric_save_plotter(0.5, 0.5, resolution, show=show, plot_letter=letter).flatten()
 
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])        
    df.insert(0, 'shape', 'circle')
    return df


# =============================================================================    
# Generator functions for multiple datasets
# =============================================================================   

# OLD Methods for purely (x,y) coordinates without image
numerical_functions = {
    'parabola' : createParabolaNumericalDataset,
    'circle'   : createCircleNumericalDataset,
    'hyperbola': createHyperbolaNumericalDataset,
    'ellipse'  : createEllipseNumericalDataset,
}

def getNumericalDatasets(*Names, sample_count=6):
    """
    Names must be a lowerkey entry in functions
    """
    combined_dataset = pd.DataFrame()
    # Can't add in place
    dfs = []
    for shape in Names:
        # call creating function via name
        dfs.append(numerical_functions[shape.lower()](sample_count))
    combined_dataset = pd.concat(dfs)
    return combined_dataset


def createSingleGrayscaleDataset(generator, name, amount=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION):
    data = np.empty((amount, np.prod(resolution)), dtype='uint8')

    for i, (x, y) in enumerate(generator):
        # Hyperbolas need some special treatmeant
        data[i] = numeric_save_plotter(x, y, resolution, plot_hyperbola=name=='hyperbola').flatten()
    
    df = pd.DataFrame(data, columns=["x_" + str(i) for i in range(np.prod(resolution))])
    df.insert(0, 'shape', name)
    return df

def createGrayscaleImageDataset(generators, names, subset_size=DEFAULT_SUBSET_SIZE, resolution=DEFAULT_RESOLUTION):
    assert len(generators) == len(names), "Amount of generator objects and amount of provides names does not match."
    dataset = pd.concat([
        createSingleGrayscaleDataset(gen, name, amount=subset_size, resolution=resolution)
        for gen, name in zip(generators, names)
    ])
    return dataset


# Activate manually or possible via command line option.
def _parse_args(argv=None):
    """Parse command line arguments (supports legacy positional style)."""
    parser = argparse.ArgumentParser(description="Generate grayscale datasets of geometric shapes.")
    # Positional legacy: run, subset_size, outfile
    parser.add_argument('action', nargs='?', choices=['run'], help="Action: use 'run' to generate dataset")
    parser.add_argument('subset_legacy', nargs='?', type=int, help="(legacy) subset size positional")
    parser.add_argument('outfile_legacy', nargs='?', help="(legacy) output filename positional")

    # Modern options
    parser.add_argument('--subset', '-n', type=int, default=None, help=f"Number of samples PER SHAPE (default {DEFAULT_SUBSET_SIZE})")
    parser.add_argument('--out', '-o', type=str, default=None, help="Output file name (.csv or .npz). Default ShapesV4.csv")
    parser.add_argument('--seed', type=int, default=None, help="Seed for RNG to make generation deterministic")
    parser.add_argument('--shapes', type=str, nargs='*', default=None,
                        help="Subset of shapes to include (triangle circle ellipse parabola hyperbola)")
    parser.add_argument('--resolution', type=int, nargs=2, metavar=('H', 'W'), default=[28, 28],
                        help="Resolution, default 28 28")
    return parser.parse_args(argv)


def _build_generators(selected_shapes, subset_size):
    gens = []
    names = []
    for shape in selected_shapes:
        s = shape.lower()
        if s == 'triangle':
            gens.append(TriangleGenerator(amount=subset_size))
            names.append('triangle')
        elif s == 'circle':
            gens.append(CircleGenerator(amount=subset_size))
            names.append('circle')
        elif s == 'ellipse':
            gens.append(EllipseGenerator(amount=subset_size))
            names.append('ellipse')
        elif s == 'parabola':
            gens.append(ParabolaGenerator(amount=subset_size))
            names.append('parabola')
        elif s == 'hyperbola':
            gens.append(HyperbolaGenerator(amount=subset_size))
            names.append('hyperbola')
        else:
            raise ValueError(f"Unknown shape '{shape}'")
    return gens, names


def _save_dataset(df: pd.DataFrame, out_file: str, resolution=DEFAULT_RESOLUTION):
    if out_file.lower().endswith('.npz'):
        # Extract pixel columns (start with x_)
        pixel_cols = [c for c in df.columns if c.startswith('x_')]
        X_flat = df[pixel_cols].to_numpy(dtype='uint8')
        h, w = resolution
        try:
            X = X_flat.reshape((-1, h, w))
        except ValueError:
            # Fallback: infer square if mismatch
            side = int(math.sqrt(X_flat.shape[1]))
            X = X_flat.reshape((-1, side, side))
        # Save labels as fixed-width unicode to avoid pickle requirement on load
        y = df['shape'].to_numpy(dtype='U16')
        np.savez_compressed(out_file, X=X, y=y)
    else:
        df.to_csv(out_file, compression='gzip', index=None)


def main(argv=None):
    args = _parse_args(argv)

    # Determine if we should run (legacy: action == 'run')
    if not args.action:
        # If no action given, do nothing (user may import library)
        return

    subset_size = args.subset if args.subset is not None else (args.subset_legacy if args.subset_legacy else DEFAULT_SUBSET_SIZE)
    out_file = args.out if args.out else (args.outfile_legacy if args.outfile_legacy else 'ShapesV4.csv')
    # Resolution is always taken from CLI (with default 28x28)
    resolution = tuple(args.resolution)

    set_seed(args.seed)

    all_shapes = ['triangle', 'circle', 'ellipse', 'parabola', 'hyperbola']
    selected_shapes = args.shapes if args.shapes else all_shapes

    gens, names = _build_generators(selected_shapes, subset_size)
    df = createGrayscaleImageDataset(generators=gens, names=names, subset_size=subset_size, resolution=resolution)
    _save_dataset(df, out_file, resolution=resolution)


if __name__ == '__main__':
    main()

