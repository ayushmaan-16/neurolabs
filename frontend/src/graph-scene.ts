import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { type Parameters, type Run } from "./api";

export type GraphStage = {
  id: string;
  title: string;
  output: string;
  preactivation?: string;
  width: number;
  weight?: string;
  bias?: string;
  relu: boolean;
};
export type CameraPose = { position: number[]; target: number[] };
export type GraphState = {
  trace: Run | null;
  run: Run | null;
  progress: number;
  selected: string;
  neuron: number;
  selectionKind: "neuron" | "edge";
  selection: { tensorId: string; index: number[] };
  onSelect: (id: string, neuron: number) => void;
  onParameterSelect: (tensorId: string, index: number[]) => void;
};

export function createGraphScene({
  mount,
  stages,
  parameters,
  state,
  targets,
  pose,
  savePose,
}: {
  mount: HTMLDivElement;
  stages: GraphStage[];
  parameters: Parameters;
  state: () => GraphState;
  targets: Map<string, HTMLElement>;
  pose: CameraPose | null;
  savePose: (pose: CameraPose) => void;
}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x030409, 1);
  renderer.domElement.setAttribute(
    "aria-label",
    "Interactive 3D graph of neurons and connections",
  );
  mount.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 120);
  const spacing = 4.2;
  const center = ((stages.length - 1) * spacing) / 2;
  const controls = new OrbitControls(camera, renderer.domElement);
  const reset = () => {
    camera.position.set(center + 1, 2.7, Math.max(12, stages.length * 3.9));
    controls.target.set(center, 0, 0);
    controls.update();
  };
  reset();
  if (pose) {
    camera.position.fromArray(pose.position);
    controls.target.fromArray(pose.target);
    controls.update();
  }
  controls.enableDamping = true;
  controls.minDistance = 3;
  controls.maxDistance = 55;
  scene.add(new THREE.AmbientLight(0xd7dff0, 1.4));
  const light = new THREE.DirectionalLight(0xa3b4d3, 1.3);
  light.position.set(center - 4, 6, 8);
  scene.add(light);
  const geometry = new THREE.SphereGeometry(0.39, 32, 24);
  const particleGeometry = new THREE.SphereGeometry(0.045, 8, 8);
  const nodes = stages.map((stage, layer) =>
    Array.from({ length: stage.width }, (_, index) => {
      const component = new THREE.Group();
      component.name = `neuron:${stage.id}:${index}`;
      component.position.set(
        layer * spacing,
        ((stage.width - 1) / 2 - index) * 1.6,
        stage.width > 2 ? (index % 2 ? 0.35 : -0.35) : 0,
      );
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshLambertMaterial({ color: 0x667da3 }),
      );
      component.add(mesh);
      if (layer === 0 || layer === stages.length - 1) {
        component.add(
          new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(layer === 0 ? -1.3 : 0.52, 0, 0),
            0.78,
            0x9aa9c6,
            0.18,
            0.12,
          ),
        );
      }
      scene.add(component);
      return { component, mesh, layer, index };
    }),
  );
  const flattenedNodes = nodes.flat();
  const segments = 40;
  const edges = stages.flatMap((stage, layer) => {
    if (!layer || !stage.weight) return [];
    const weights = parameters.tensors[stage.weight].values as number[][];
    return weights.flatMap((row, target) =>
      row.map((weight, source) => {
        const component = new THREE.Group();
        component.name = `connection:${stage.weight}:${target}:${source}`;
        const from = nodes[layer - 1][source].component.position;
        const to = nodes[layer][target].component.position;
        component.position.lerpVectors(from, to, 0.5);
        const direction = to.clone().sub(from).normalize();
        const start = from
          .clone()
          .sub(component.position)
          .addScaledVector(direction, 0.4);
        const end = to
          .clone()
          .sub(component.position)
          .addScaledVector(direction, -0.4);
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(),
          new THREE.Vector3(),
          end,
        );
        const buffer = new THREE.BufferGeometry();
        buffer.setAttribute(
          "position",
          new THREE.BufferAttribute(new Float32Array((segments + 1) * 3), 3),
        );
        const line = new THREE.Line(
          buffer,
          new THREE.LineBasicMaterial({
            color: 0x9aafd9,
            transparent: true,
            opacity: 0.45,
          }),
        );
        const dot = new THREE.Mesh(
          particleGeometry,
          new THREE.MeshBasicMaterial({ color: 0xb4c5e9 }),
        );
        dot.visible = false;
        component.add(line, dot);
        scene.add(component);
        return {
          component,
          curve,
          line,
          dot,
          layer,
          row: target,
          col: source,
          weight,
          phase: layer * 1.7 + target * 1.13 + source * 0.71,
          action: () =>
            state().onParameterSelect(stage.weight!, [target, source]),
        };
      }),
    );
  });
  const raycaster = new THREE.Raycaster();
  let down = { x: 0, y: 0 };
  const hit = (event: PointerEvent) => {
    const bounds = renderer.domElement.getBoundingClientRect();
    raycaster.params.Line = {
      threshold: Math.max(
        0.04,
        ((camera.position.distanceTo(controls.target) *
          Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) *
          2) /
          bounds.height) *
          5,
      ),
    };
    raycaster.setFromCamera(
      new THREE.Vector2(
        ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
        (-(event.clientY - bounds.top) / bounds.height) * 2 + 1,
      ),
      camera,
    );
    const nodeHit = raycaster.intersectObjects(
      flattenedNodes.map((item) => item.mesh),
    )[0];
    if (nodeHit) {
      const node = flattenedNodes.find((item) => item.mesh === nodeHit.object)!;
      return () => state().onSelect(stages[node.layer].id, node.index);
    }
    const edgeHit = raycaster.intersectObjects(
      edges.map((edge) => edge.line),
    )[0];
    return edgeHit
      ? edges.find((edge) => edge.line === edgeHit.object)!.action
      : undefined;
  };
  const pointerDown = (event: PointerEvent) => {
    down = { x: event.clientX, y: event.clientY };
  };
  const pointerUp = (event: PointerEvent) => {
    if (
      event.button === 0 &&
      Math.hypot(event.clientX - down.x, event.clientY - down.y) <= 5
    )
      hit(event)?.();
  };
  const pointerMove = (event: PointerEvent) => {
    renderer.domElement.style.cursor = event.buttons
      ? "grabbing"
      : hit(event)
        ? "pointer"
        : "grab";
  };
  renderer.domElement.addEventListener("pointerdown", pointerDown);
  renderer.domElement.addEventListener("pointerup", pointerUp);
  renderer.domElement.addEventListener("pointermove", pointerMove);
  const resize = new ResizeObserver(() => {
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    camera.aspect = mount.clientWidth / mount.clientHeight;
    camera.updateProjectionMatrix();
  });
  resize.observe(mount);
  const world = new THREE.Vector3(),
    right = new THREE.Vector3(),
    up = new THREE.Vector3();
  const corner1 = new THREE.Vector3(),
    corner2 = new THREE.Vector3();
  // Invisible keyboard targets follow the actual node or curve; the scene contains no text.
  const syncTarget = (
    position: THREE.Vector3,
    width: number,
    height: number,
    key: string,
  ) => {
    const element = targets.get(key);
    if (!element) return;
    right
      .set(1, 0, 0)
      .applyQuaternion(camera.quaternion)
      .multiplyScalar(width / 2);
    up.set(0, 1, 0)
      .applyQuaternion(camera.quaternion)
      .multiplyScalar(height / 2);
    corner1.copy(position).sub(right).add(up).project(camera);
    corner2.copy(position).add(right).sub(up).project(camera);
    const x = ((corner1.x + 1) * mount.clientWidth) / 2,
      y = ((1 - corner1.y) * mount.clientHeight) / 2;
    const w = ((corner2.x - corner1.x) * mount.clientWidth) / 2,
      h = ((corner1.y - corner2.y) * mount.clientHeight) / 2;
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    element.style.width = `${Math.max(0, w)}px`;
    element.style.height = `${Math.max(0, h)}px`;
    element.style.visibility =
      corner1.z > 1 ||
      x + w < 0 ||
      x > mount.clientWidth ||
      y + h < 0 ||
      y > mount.clientHeight
        ? "hidden"
        : "visible";
  };
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const started = performance.now();
  const point = new THREE.Vector3();
  renderer.setAnimationLoop(() => {
    controls.update();
    const current = state();
    const values = stages.map(
      (stage) =>
        (current.trace ?? current.run)?.activations[stage.output]?.[0] ?? [],
    );
    const magnitude = Math.max(1, ...values.flat().map(Math.abs));
    flattenedNodes.forEach((node) => {
      const reached = current.trace
        ? current.progress >= node.layer
        : !!current.run;
      const value = values[node.layer][node.index] ?? 0;
      const active = reached && value !== 0;
      node.mesh.material.color.setHex(
        active ? (value < 0 ? 0xd5a995 : 0x9aafd9) : 0x252b3b,
      );
      node.mesh.material.emissive.copy(node.mesh.material.color);
      node.mesh.material.emissiveIntensity = active
        ? 0.08 + (Math.abs(value) / magnitude) * 0.12
        : 0;
      node.component.scale.setScalar(
        current.selectionKind === "neuron" &&
          stages[node.layer].id === current.selected &&
          node.index === current.neuron
          ? 1.12
          : 1,
      );
    });
    const time = reducedMotion.matches
      ? 0
      : (performance.now() - started) / 1000;
    edges.forEach((edge) => {
      // Endpoints stay attached; only the two interior control points gently drift.
      // This is a visual string deformation, never a change to model parameters or computation.
      const sway = Math.sin(time * 0.65 + edge.phase) * 0.14;
      const depth = Math.cos(time * 0.48 + edge.phase) * 0.22;
      edge.curve.v1.lerpVectors(edge.curve.v0, edge.curve.v3, 0.33);
      edge.curve.v1.y += -0.5 + sway;
      edge.curve.v1.z += 0.3 + depth;
      edge.curve.v2.lerpVectors(edge.curve.v0, edge.curve.v3, 0.67);
      edge.curve.v2.y += -0.34 - sway * 0.6;
      edge.curve.v2.z += -0.2 + depth;
      const positions = edge.line.geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;
      for (let i = 0; i <= segments; i++) {
        edge.curve.getPoint(i / segments, point);
        positions.setXYZ(i, point.x, point.y, point.z);
      }
      positions.needsUpdate = true;
      edge.line.geometry.computeBoundingSphere();
      const contribution =
        (values[edge.layer - 1][edge.col] ?? 0) * edge.weight;
      const reached = current.trace
        ? current.progress >= edge.layer - 1
        : !!current.run;
      const picked =
        current.selectionKind === "edge" &&
        current.selection.tensorId === stages[edge.layer].weight &&
        current.selection.index[0] === edge.row &&
        current.selection.index[1] === edge.col;
      edge.line.material.color.setHex(
        picked
          ? 0xe1eaff
          : reached
            ? contribution < 0
              ? 0xd5a995
              : 0x9aafd9
            : 0x52627c,
      );
      edge.line.material.opacity = picked
        ? 1
        : reached && contribution !== 0
          ? 0.6
          : 0.3;
      edge.dot.material.color.copy(edge.line.material.color);
      edge.dot.visible =
        !!current.trace &&
        reached &&
        contribution !== 0 &&
        current.progress < edge.layer;
      if (edge.dot.visible)
        edge.dot.position.copy(
          edge.curve.getPoint(current.progress - edge.layer + 1),
        );
    });
    renderer.render(scene, camera);
    flattenedNodes.forEach((node) => {
      node.mesh.getWorldPosition(world);
      syncTarget(
        world,
        0.78 * node.component.scale.x,
        0.78 * node.component.scale.x,
        `${node.layer}:${node.index}`,
      );
    });
    edges.forEach((edge) => {
      world.copy(edge.curve.getPoint(0.46)).add(edge.component.position);
      syncTarget(
        world,
        0.22,
        0.22,
        `edge:${edge.layer}:${edge.row}:${edge.col}`,
      );
    });
  });
  return {
    reset,
    dispose: () => {
      savePose({
        position: camera.position.toArray(),
        target: controls.target.toArray(),
      });
      renderer.setAnimationLoop(null);
      resize.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("pointermove", pointerMove);
      const geometries = new Set<THREE.BufferGeometry>(),
        materials = new Set<THREE.Material>();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          geometries.add(object.geometry);
          (Array.isArray(object.material)
            ? object.material
            : [object.material]
          ).forEach((material) => materials.add(material));
        }
      });
      geometries.forEach((item) => item.dispose());
      materials.forEach((item) => item.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
