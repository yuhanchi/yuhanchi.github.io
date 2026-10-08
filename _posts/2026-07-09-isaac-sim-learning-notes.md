---
title: "An Uphill Journey with Isaac Sim in a Downhill Lab"
date: "2026-07-09"
description: "Learning Isaac Sim at Sutardja Dai Hall. The lab is downhill, but this learning curve is definitely uphill."
category: "tech"
tags: ["Isaac Sim", "Simulation", "Robotics"]
---
## Glossary

This section provides an explanation of the terms used throughout NVIDIA Isaac Sim and replicates several of the terms defined in the Omniverse Glossary.

### Omniverse

**Application / Apps**
An Omniverse App is built upon a specific set of Extensions to provide a desired functionality. An App gives the user a customized experience by implementing the UI’s of its Extensions with a custom layout. You can quickly and easily create customized Apps comprised of any number of Extensions developed by you, the Omniverse Community or NVIDIA. An App can be as simple as a 3D viewer or as complex as an AI suite. This modular approach to building Apps makes it easy to create a customized workflow or a global scale cloud application.

**Connectors**
An Omniverse Connector is middleware with which Omniverse and other software applications communicate with each other. They enable the import/export 3D assets, data, and models between different tools and workflows. It’s important to note that this means using USD as the “go between” format to convert 3D data.

**Omniverse Nucleus**
Omniverse Nucleus offers a set of fundamental services that allow a variety of client applications, renderers, and microservices to share and modify representations of virtual worlds.
Nucleus operates under a publish/subscribe model. Subject to access controls, Omniverse clients can publish modifications to digital assets and virtual worlds to the Nucleus Database (DB) or subscribe to their changes. Changes are transmitted in real-time between connected applications. Digital assets can include geometry, lights, materials, textures and other data that describe virtual worlds and their evolution through time.
This allows a variety of Omniverse-enabled client applications (Apps, Connectors, and others) to share and modify authoritative representations of virtual worlds.

**Hub Workstation Cache**
Hub Workstation Cache is a service that helps speed up USD workflows on your local workstation. This is a stand-alone service that runs on your local workstation and benefits Kit-based applications or Client Library tools.

**Live Sync**
Live Sync mode enables real-time “live” editing of shared files on a Nucleus Server. The Live Sync button is on the top-right corner of the Workspace.

**Omniverse Kit**
NVIDIA Omniverse™ Kit is a toolkit for building native Omniverse applications and microservices. It is built on a base framework known as Carbonite that provides a wide variety of functionality through a set of light-weight plugins. Carbonite plugins are all authored with C interfaces for persistent ABI compatibility. A Python interpreter is provided for scripting and customization.

**Omniverse Launcher**
The NVIDIA Omniverse Launcher is your first step into the Omniverse. It provides immediate access to all the apps, connectors and other downloads within the Omniverse.

**Omniverse USD Composer**
NVIDIA Omniverse™ USD Composer was an Omniverse app for world-building that allows users to assemble, light, simulate and render large scale scenes. It is built using NVIDIA Omniverse™ Kit. The Scene Description and in-memory model is based on Pixar’s USD. USD Composer takes advantage of the advanced workflows of USD like Layers, Variants, Instancing and much more.

**Carbonite (carb)**
The Carbonite SDK provides the core functionality of all Omniverse apps. This is a C++ based SDK with Python bindings that provides features such as plugin management, input handling, file access, asset loading and management, thread and task management, and much more.

**RTX - Real-Time mode**
High quality real-time rendering mode.

**RTX – Interactive (Path Tracing) mode**
The highest quality, physically accurate rendering mode.

**Extensions**
Extensions are plug-ins to Omniverse Kit that extend its capabilities. They are offered with complete source code to help developers easily create, add, and modify the tools and workflows they need to be productive. Extensions are the core building blocks of Omniverse Kit based applications.

**Omniverse Connect**
Connectors are extensions and additional software layers on top of the open-source USD distribution that allow DCC tools and compute services to communicate easily with each other through the Omniverse Nucleus DB. Those extensions and additions are collectively known as NVIDIA Omniverse™ Connect.

### USD

**USD (Universal Scene Description)**
Universal Scene Description (USD) is an easily extensible, open-source 3D scene description file format developed by Pixar for content creation and interchange among different tools. As a result of its power and versatility, it’s being widely adopted, not only in the visual effects community, but also in architecture, design, robotics, manufacturing, and other disciplines.

**MDL**
Material Definition Language (MDL) is a NVIDIA-developed USD schema that represents material assignments and specifies material parameters.

**Stage**
The Omniverse Stage window allows you to see all the assets in your current USD Scene. The USD Stage is the USD abstraction for a scenegraph derived from a root USD file, and all of the referenced/layered files it composes. Listed in a hierarchical (parent/child) order the Stage offers convenient access and is typically used to navigate large scenes.

**Prim**
A Prim is the primary container object in USD: prims can contain (and order) other prims, creating a “namespace hierarchy” on a Stage, and prims can also contain (and order) properties that hold meaningful data. Prims, along with their associated, computed indices, are the only persistent scenegraph objects that a Stage retains in memory, and the API for interacting with prims is provided by the UsdPrim class.

**Mesh**
A mesh is a subdividable primitive that consists of points, edges, and faces that define its shape. In USD, a mesh is encoded in a UseGeomMesh class.

**Shape**
A Shape is a geometric primitive that maps to one of USD’s five “intrinsic” UsdGeomGprim classes (Capsule, Cone, Cube, Cylinder, Sphere). Shapes are not meshes, in that they are not defined by a collection of points, edges, and faces. Instead, they are defined by their shape and volume.

**Reference vs Payload vs Instance**
Everything in USD is a primitive (prim) with attributes. Some of these primitives are defined in your current layer (the active stage), while others are defined in other layers (other USD files).
- **Reference**: Lightweight inclusion from another layer. Implies child prims won't be modified.
- **Payload**: References that have all their data actively loaded into memory so they can be modified at runtime.
- **Instance**: A light-weight and less manipulable copy of a prim (can be references or payloads), meant for efficient vectorization (scaling up).

**Y-Up / Z-Up**
The axis of orientation of a given scene/prim. Y-Up refers to the Positive Y Axis pointing up. Z-Up refers to the Positive Z Axis pointing up.

**Layer**
A component of the collaborative nature of USD. Each layer in USD signifies a user’s “opinion” on assets inside a stage. Layers can override other layers.

**Checkpoint**
Immutable historical file versions. Checkpoints are used for version control and allow you to look at and restore the stage to a previous state.

### Isaac Sim

**PhysX**
NVIDIA PhysX is a scalable multi-platform physics simulation solution. The NVIDIA Omniverse™ Physics simulation extension is powered by the NVIDIA PhysX SDK, and includes Rigid Body Simulation, Articulations, Deformable-Body Simulation, and Character Controller.

**ROS / ROS 2**
The Robot Operating System (ROS) is a set of software libraries and tools that help you build robot applications. NVIDIA Isaac Sim provides many extensions, examples, and APIs for connecting to ROS and ROS 2 workflows.

**Dynamic Control (Deprecated)**
The Dynamic Control extension is a set of utilities to control physics objects. It provides opaque handles for different physics objects that remain valid between PhysX scene resets.

**Core API**
The Isaac Core Extension in Isaac Sim provides high-level interfaces to PhysX and raw USD APIs. It abstracts away default parameters to simplify creation and manipulation of a simulated world and scenarios encountered in robotics simulators.

**Riemannian Motion Policy (RMP)**
Riemannian Motion Policy (RMP) is a set of motion generation tools that underlies most of our manipulator controls inside Omniverse Isaac Sim. It creates smooth trajectories for the robots with intelligent collision avoidance.

**World**
World is the core class that enables you to interact with the simulator in an easy and modular way. It takes care of many time-related events such as adding callbacks, stepping physics, resetting the scene, adding tasks, etc. The World class is a Singleton.

**Scene**
A world contains an instance of a Scene, think about it as a scene management class that manages the assets of interest in the USD stage.

**Task**
The Task class in isaacsim.core.api provides a way to modularize the scene creation, information retrieval, calculating metrics and creating more complex scenes with more involved logic.

**Articulation**
An articulated robot is a robot with rotary joints (e.g: a legged robot, a manipulator or a wheeled robot). The Articulation class enables interaction with articulations that exist in a USD stage.

**Replicator & Synthetic Data Generation & Ground Truth**
Replicator is a Synthetic Data Generation tool for creating parameterizable offline datasets in NVIDIA Isaac Sim. It can be used to generate ground truth data that is very similar to real-life analogs.

---
**Community Project Highlights**


My device: win11+wsl2

1. Download windows version into your windows system.
2. Unpack it. Because of the large file, it may fail. A trick is to rename your folder. It seems like the shorter you folder's name is, the more likely you unpack successfully.
3. use `isaac-sim.bat` to run the UI app. If you dislike UI (like me), you can choose `/path_to_folder/python.bat /path_to_script/your_script.py`


**Notice**

1. Administrative mode(管理员模式)
2. 链接？我忘了一开始要不要了

Note:

1. Do not try to build things in UI? By python is easier.
2. Basic element is "Prim"
3. We view all things as a tree, wo we have to find a certain "prim" like searching a tree
4. The keywords during search are tricky. I haven't figure d it out.
5. When creating things, we are acutally making an instance

```python
    cylinder = DynamicCylinder(
        prim_path="/World/Object",
        name="cylinder",
        position=cylinder_spawn_pos,
        radius=0.04,
        height=0.05,
        mass=np.random.uniform(0.01, 0.25),  # [0.01, 0.25]kg
        color=np.array([0.0, 0.0, 1.0])       # 蓝色
    )
```

6. The form of data is very important. By default, all data are stored in tensor, like 

```python
    base_pose = torch.tensor([
        # 食指 (Index) - 略微向外张开
        -0.15,  0.5,  0.6,  0.5,
        # 中指 (Middle) - 笔直向前
         0.00,  0.5,  0.6,  0.5,
        # 无名指 (Ring) - 略微向外张开 (与食指对称)
         0.15,  0.5,  0.6,  0.5,
        # 大拇指 (Thumb) - 旋转基座以对向其他三根手指
         1.00,  0.4,  0.4,  0.4
    ], dtype=torch.float32)
```
