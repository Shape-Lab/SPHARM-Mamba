"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import os
import numpy as np
from datetime import datetime


def write_vtk(fname, v, f, prop=None):
    """
    Write vtk file with vertex-wise meta data.

    Parameters
    __________
    fname : str
        Output file name.
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    prop : dict, shape = ("attribute name", vertex-wise data)
        Attribute can be any 1D scalar array with size of (n_vertex, 1).
    """

    len_v = v.shape[0]
    len_f = f.shape[0]

    with open(fname, "w") as fd:
        fd.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        fd.write(f"POINTS {len_v} float\n")
        for row in v:
            fd.write(f"{row[0]} {row[1]} {row[2]}\n")
        fd.write(f"POLYGONS {len_f} {len_f * 4}\n")
        for row in f:
            fd.write(f"3 {row[0]} {row[1]} {row[2]}\n")
        if prop is not None:
            fd.write(f"POINT_DATA {len_v}\n")
            fd.write(f"FIELD ScalarData {len(prop)}\n")
            for key in prop.keys():
                fd.write(f"{key} 1 {len_v} float\n")
                val = prop[key]
                for num in val:
                    fd.write(f"{num}\n")


def write_surf(fname, v, f, tag=""):
    """
    Write FreeSurfer's mesh file.

    Parameters
    __________
    fname : str
        Output file name.
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    tag : str
        String for metadata.
    """

    with open(fname, "wb") as fd:
        fd.write(0xFFFFFE.to_bytes(3, byteorder="big"))
        fd.write(f"created by {os.environ['USER']} on {datetime.now().ctime()}\n\n".encode())
        fd.write(np.asarray([v.shape[0], f.shape[0]], dtype=">i4").tobytes())
        fd.write(v.astype(">f4").tobytes())
        fd.write(f.astype(">i4").tobytes())
        if tag != "":
            if isinstance(tag, str):
                tag = [tag]
            for t in tag:
                fd.write(0x03.to_bytes(4, byteorder="big"))
                fd.write(len(t).to_bytes(8, byteorder="big"))
                fd.write(t.encode())


def write_mesh(fname, v, f, tag=""):
    """
    Write a mesh file.

    Parameters
    __________
    fname : str
        Output file name.
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    tag : str
        String for metadata.
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".vtk":
        write_vtk(fname, v, f)
    else:
        write_surf(fname, v, f, tag)
