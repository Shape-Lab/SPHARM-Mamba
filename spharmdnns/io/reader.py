"""
July 2021

Seungbo Ha, mj0829@unist.ac.kr
Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import os
import numpy as np
from scipy.interpolate import interpn
from PIL import Image


def read_curv(fname, nvert=-1):
    """
    Read FreeSurfer's geometry.

    Parameters
    __________
    fname : str
        File path.
    nvert : int
        Number of vertices to be read.

    Returns
    _______
    feat : 1D array
        Vertex-wise geometry.
    fnum : int
        Number of faces.

    Notes
    _____
        https://github.com/fieldtrip/fieldtrip/tree/master/external/freesurfer
        https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_curv.m
        They read binary files in big-endian.
    """

    with open(fname, "rb") as fd:
        h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
        vnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

        if vnum == 0xFFFFFF:
            vnum, fnum, vals_per_vertex = np.fromfile(fd, dtype=np.dtype(">i4"), count=3)
            feat = np.fromfile(fd, dtype=np.dtype(">f4"), count=nvert)
        else:
            f0, f1, f2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            fnum = (f0 << 16) + (f1 << 8) + f2
            feat = np.fromfile(fd, dtype=np.dtype(">i2"), count=nvert) / 100

    return feat, fnum


def read_surf(fname):
    """
    Read FreeSurfer's surface.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vertex_coords : 2D array, shape = [n_vertex, 3]
        Vertex coordinates.
    faces : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    with open(fname, "rb") as fd:
        h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
        magic = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

        if (magic == 0xFFFFFF) | (magic == 0xFFFFFD):
            # need to be verified
            h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            vnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

            h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            fnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

            vertex_coords = np.fromfile(fd, dtype=np.dtype(">i2"), count=3 * vnum) / 100
            vertex_coords = vertex_coords.reshape(-1, 3)
            arr = np.fromfile(fd, dtype=np.dtype("B"), count=12 * fnum).astype(int)
            faces = (arr[0::3] << 16) + (arr[1::3] << 8) + arr[2::3]
            faces = faces.reshape(-1, 4)

            return vertex_coords.astype(float), faces.astype(int)

        elif magic == 0xFFFFFE:
            fd.readline()
            fd.readline().strip()

            vnum, fnum = np.fromfile(fd, dtype=np.dtype(">i4"), count=2)
            vertex_coords = np.fromfile(fd, dtype=np.dtype(">f4"), count=3 * vnum)
            faces = np.fromfile(fd, dtype=np.dtype(">i4"), count=3 * fnum)

            vertex_coords = vertex_coords.reshape(vnum, 3)
            faces = faces.reshape(fnum, 3)

            return vertex_coords.astype(float), faces.astype(int)

    raise Exception("SurfReaderError: unknown format!")


def read_dat(fname, nvert=-1):
    """
    Read SPHARM-Net's geometry (.dat)

    - *.label.dat  -> int16   (segmentation label)
    - *.float.dat  -> float64 (scalar age label)
    - others       -> float32 (features: sulc/curv/thickness/area/inflated.H ...)
    """
    attr = fname.lower().split(".")[-2]  # e.g. 'sulc', 'curv', 'thickness', 'label', 'float'

    if attr == "label":
        dtype = np.int16
    elif attr == "float":
        dtype = np.float64
    else:
        dtype = np.float64   # ✅ 핵심

    return np.fromfile(fname, dtype=dtype, count=nvert)

def read_txt(fname, nvert=-1):
    """
    Read geometry.

    Parameters
    __________
    fname : str
        File path.
    nvert : int
        Number of vertices to be read.

    Returns
    _______
    feat : 1D array
        Vertex-wise geometry.
    """

    return np.fromfile(fname, dtype=np.float64, sep=" ", count=nvert)


def read_annot(fname):
    """
    Read FreeSurfer's annot.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vID : 1D array
        Vertex IDs.
    label : 1D array
        Vertex-wise label.
    struct : 1D array.
        Structure name.
    structID : 1D array.
        Structure ID.
    """

    with open(fname, "rb") as fd:
        size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
        buf = np.fromfile(fd, dtype=np.dtype(">i4"), count=2 * size).reshape(-1, 2)
        vID, label = buf[:, 0], buf[:, 1]

        struct = []
        structID = []

        if np.fromfile(fd, dtype=np.dtype(">i4"), count=1).astype(bool)[0]:
            nEntries = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]

            version = 0
            if nEntries <= 0:
                version = -nEntries
                if version != 2:
                    raise Exception("AnnotReaderError: version != 2", version)
                nEntries = np.fromfile(fd, dtype=">i4", count=1)

            size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
            buf = np.fromfile(fd, dtype=np.dtype(">i1"), count=size)
            # orig_tab = buf[:-1].tobytes().decode("ascii")

            if version == 2:
                nEntries = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]

            for _ in range(nEntries):
                if version == 2 and np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0] < 0:
                    raise Exception("AnnotReaderError: entry index < 0")

                size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
                buf = np.fromfile(fd, dtype=np.dtype(">i1"), count=size)
                struct.append(buf[:-1].tobytes().decode("ascii"))
                buf = np.fromfile(fd, dtype=np.dtype(">i4"), count=4)
                structID.append(buf[0] + (buf[1] << 8) + (buf[2] << 16))

    return vID.astype(int), label.astype(int), struct, structID


def read_feat(fname, nvert=-1):
    """
    Read a geometry file.
    Parameters
    __________
    fname : str
        File path.
    nvert : int
        Number of vertices to be read.

    Returns
    _______
    feat : 1D array
        Vertex-wise geometry.
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".txt":
        feat = read_txt(fname, nvert)
    elif ext == ".dat":
        feat = read_dat(fname, nvert)
    else:
        feat, _ = read_curv(fname, nvert)

    return feat


def read_vtk(fname):
    """
    Read a vtk file (ASCII version and VTK < v4.0).

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    with open(fname, "rb") as fd:
        lines = iter(l for l in fd)

        ver = next(d for d in lines if b"Version" in d)
        ver = float(ver.split()[-1])
        bin = next(d.strip() for d in lines if d.strip() in [b"ASCII", b"BINARY"]) == b"BINARY"
        sep = "" if bin else " "

        nVert = next(d for d in lines if b"POINTS" in d)
        nVert = int(nVert.split()[1])
        dtype = np.dtype(">f4") if bin else float
        v = np.fromfile(fd, dtype=dtype, count=nVert * 3, sep=sep).reshape(nVert, 3)

        nFace = next(d for d in lines if b"POLYGONS" in d)
        nFace = int(nFace.split()[1])
        if ver < 5:
            dtype = np.dtype(">i4") if bin else int
            f = np.fromfile(fd, dtype=dtype, count=nFace * 4, sep=sep).reshape(nFace, 4)
            f = f[:, 1:]
        else:
            dtype = np.dtype(">i8") if bin else int
            nFace -= 1
            next(d for d in lines if b"CONNECTIVITY" in d)
            f = np.fromfile(fd, dtype=dtype, count=nFace * 3, sep=sep).reshape(nFace, 3)

    return v, f


def read_mesh(fname):
    """
    Read a mesh file.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".vtk":
        v, f = read_vtk(fname)
    else:
        v, f = read_surf(fname)

    return v, f


def read_tif(fname, v):
    """
    Read FreeSurfer's templates.

    Parameters
    __________
    fname : str
        File path.
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.

    Returns
    _______
    mean : a dictionary of 1D arrays
        Vertex-wise mean of geometric features: inflated.H, sulc, curv.
    var : a dictionary of 1D arrays
        Vertex-wise variance of geometric features: inflated.H, sulc, curv.
    dof : a dictionary of scalars
        Degree of freedom: inflated.H, sulc, curv.
    """

    keys = ["inflated.H", "sulc", "curv"]
    mean, var, dof = {}, {}, {}
    targets = [mean, var, dof]

    im = Image.open(fname)
    w, h = im.size
    x = np.linspace(-np.pi / 2, np.pi / 2, w + 1)
    y = np.linspace(-np.pi, np.pi, h + 1)
    points = (y, x)
    a = np.arctan2(v[:, 1], -v[:, 0])
    e = -np.arctan2(v[:, 2], np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2))
    point = np.hstack((a[:, None], e[:, None]))
    for frame in range(9):
        im.seek(frame)
        values = np.asarray(im).tobytes()
        values = np.frombuffer(values, dtype=np.float32).reshape((h, w)).astype(float)
        values = np.vstack((values[-1:], values))
        values = np.hstack((values, values[:, -1:]))

        key = keys[frame // 3]
        target = targets[frame % 3]
        target[key] = values[0, 0].astype(int) if frame % 3 == 2 else interpn(points, values, point, method="linear")

    return mean, var, dof
