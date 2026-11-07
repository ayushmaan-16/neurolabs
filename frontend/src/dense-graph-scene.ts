import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraPose, GraphStage, GraphState } from "./graph-scene";
import type { GraphConnection, GraphDetail } from "./graph-detail";

// One instanced neuron mesh, one batched string mesh, one instanced particle mesh.
// Draw calls stay bounded as layer width grows; hidden connections are never simulated away.
export function createDenseGraphScene({
  mount,
  stages,
  connections,
  detail,
  focusLayer,
  selectedLayer,
  state,
  targets,
  pose,
  savePose,
}: {
  mount: HTMLDivElement;
  stages: GraphStage[];
  connections: GraphConnection[];
  detail: GraphDetail;
  focusLayer: number;
  selectedLayer: number;
  state: () => GraphState;
  targets: Map<string, HTMLElement>;
  pose: CameraPose | null;
  savePose: (pose: CameraPose) => void;
}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setClearColor(0x030409, 1);
  renderer.domElement.setAttribute(
    "aria-label",
    "Interactive 3D graph of neurons and connections",
  );
  mount.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  let viewportWidth = mount.clientWidth,
    viewportHeight = mount.clientHeight;
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 2;
  controls.maxDistance = 900;
  scene.add(new THREE.AmbientLight(0xd7dff0, 1.5));
  const light = new THREE.DirectionalLight(0xa3b4d3, 1.4);
  light.position.set(15, 20, 40);
  scene.add(light);
  const nodes = stages.flatMap((stage, layer) => {
    const columns = stage.width > 8 ? Math.ceil(Math.sqrt(stage.width)) : 1;
    const rows = Math.ceil(stage.width / columns);
    return Array.from({ length: stage.width }, (_, index) => ({
      layer,
      index,
      position: new THREE.Vector3(
        layer * 7,
        ((rows - 1) / 2 - Math.floor(index / columns)) * 1.05,
        ((index % columns) - (columns - 1) / 2) * 1.05,
      ),
    }));
  });
  const relevantLayer = (layer: number) =>
    detail === "overview" ||
    (detail === "layer"
      ? layer === focusLayer || layer === focusLayer - 1
      : Math.abs(layer - selectedLayer) <= 1);
  const byLayer = stages.map((_, layer) =>
    nodes.filter((node) => node.layer === layer),
  );
  const nodeGeometry = new THREE.SphereGeometry(0.24, 12, 10);
  const nodeMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const neurons = new THREE.InstancedMesh(
    nodeGeometry,
    nodeMaterial,
    nodes.length,
  );
  neurons.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const matrix = new THREE.Matrix4();
  nodes.forEach((node, index) => {
    neurons.setMatrixAt(index, matrix.makeTranslation(node.position));
    neurons.setColorAt(index, new THREE.Color(0x667da3));
    if (
      relevantLayer(node.layer) &&
      (node.layer === 0 || node.layer === stages.length - 1)
    ) {
      scene.add(
        new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          node.position
            .clone()
            .add(new THREE.Vector3(node.layer === 0 ? -1.2 : 0.4, 0, 0)),
          0.75,
          0x9aa9c6,
          0.2,
          0.13,
        ),
      );
    }
  });
  scene.add(neurons);

  const segments = 12;
  const positions = new Float32Array(connections.length * segments * 6);
  const colors = new Float32Array(positions.length);
  const edgeGeometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3).setUsage(
    THREE.DynamicDrawUsage,
  );
  const colorAttribute = new THREE.BufferAttribute(colors, 3).setUsage(
    THREE.DynamicDrawUsage,
  );
  edgeGeometry.setAttribute("position", positionAttribute);
  edgeGeometry.setAttribute("color", colorAttribute);
  const edgeMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.62,
  });
  const strings = new THREE.LineSegments(edgeGeometry, edgeMaterial);
  strings.frustumCulled = false;
  scene.add(strings);
  const particleGeometry = new THREE.SphereGeometry(0.045, 6, 6);
  const particleMaterial = new THREE.MeshBasicMaterial({ color: 0xc1cfee });
  const particles = new THREE.InstancedMesh(
    particleGeometry,
    particleMaterial,
    Math.max(1, connections.length),
  );
  particles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  particles.frustumCulled = false;
  scene.add(particles);
  const edges = connections.map((edge) => {
    const from = byLayer[edge.layer - 1][edge.col].position;
    const to = byLayer[edge.layer][edge.row].position;
    const direction = to.clone().sub(from).normalize();
    return {
      ...edge,
      phase: edge.layer * 1.7 + edge.row * 1.13 + edge.col * 0.71,
      curve: new THREE.CubicBezierCurve3(
        from.clone().addScaledVector(direction, 0.25),
        new THREE.Vector3(),
        new THREE.Vector3(),
        to.clone().addScaledVector(direction, -0.25),
      ),
    };
  });
  const fit = (focused: boolean) => {
    const layer = detail === "layer" ? focusLayer : selectedLayer;
    const subset = focused
      ? nodes.filter(
          (node) =>
            Math.abs(node.layer - layer) <= (detail === "layer" ? 0 : 1) ||
            (detail === "layer" && node.layer === layer - 1),
        )
      : nodes;
    const box = new THREE.Box3().setFromPoints(
      subset.map((node) => node.position),
    );
    box.expandByScalar(1.2);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const distance =
      Math.max(size.y, size.x / camera.aspect) /
        (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) +
      size.z +
      4;
    controls.target.copy(center);
    camera.position
      .copy(center)
      .add(
        new THREE.Vector3(
          focused ? 10 : distance * 0.6,
          focused ? 6 : distance * 0.45,
          distance,
        ),
      );
    controls.update();
  };
  const reset = () => fit(detail !== "overview");
  let initial = true;
  const resize = new ResizeObserver(() => {
    viewportWidth = mount.clientWidth;
    viewportHeight = mount.clientHeight;
    renderer.setSize(viewportWidth, viewportHeight);
    camera.aspect = viewportWidth / Math.max(1, viewportHeight);
    camera.updateProjectionMatrix();
    if (initial) {
      if (pose) {
        camera.position.fromArray(pose.position);
        controls.target.fromArray(pose.target);
        controls.update();
      } else reset();
      initial = false;
    }
  });
  resize.observe(mount);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pick = (event: PointerEvent) => {
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      (-(event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const nodeHit = raycaster.intersectObject(neurons)[0];
    if (nodeHit?.instanceId !== undefined) {
      const node = nodes[nodeHit.instanceId];
      return () => state().onSelect(stages[node.layer].id, node.index);
    }
    raycaster.params.Line = {
      threshold: Math.max(
        0.035,
        (camera.position.distanceTo(controls.target) *
          Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) *
          8) /
          bounds.height,
      ),
    };
    // Dense bundles often have several strings inside the picking tolerance.
    // Choose the closest visible stroke in screen space, rather than the first
    // (nearest-camera) intersection, which can select a neighboring string.
    let bestDistance = Infinity;
    let hit: THREE.Intersection | undefined;
    const screenPoint = new THREE.Vector3();
    for (const candidate of raycaster.intersectObject(strings)) {
      screenPoint.copy(candidate.point).project(camera);
      const distance = Math.hypot(
        (screenPoint.x - pointer.x) * bounds.width,
        (screenPoint.y - pointer.y) * bounds.height,
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        hit = candidate;
      }
    }
    const edge =
      hit?.index !== undefined
        ? edges[Math.floor(hit.index / (segments * 2))]
        : undefined;
    return edge
      ? () => state().onParameterSelect(edge.tensorId, [edge.row, edge.col])
      : undefined;
  };
  let down = { x: 0, y: 0 };
  const pointerDown = (event: PointerEvent) => {
    down = { x: event.clientX, y: event.clientY };
  };
  const pointerUp = (event: PointerEvent) => {
    if (
      event.button === 0 &&
      Math.hypot(event.clientX - down.x, event.clientY - down.y) <= 5
    )
      pick(event)?.();
  };
  renderer.domElement.style.cursor = "grab";
  renderer.domElement.addEventListener("pointerdown", pointerDown);
  renderer.domElement.addEventListener("pointerup", pointerUp);

  const projected = new THREE.Vector3();
  const syncTarget = (
    position: THREE.Vector3,
    diameter: number,
    key: string,
  ) => {
    const element = targets.get(key);
    if (!element) return;
    projected.copy(position).project(camera);
    const distance = position.distanceTo(camera.position);
    const pixels =
      (diameter * viewportHeight) /
      (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * distance);
    element.style.left = `${((projected.x + 1) * viewportWidth) / 2 - pixels / 2}px`;
    element.style.top = `${((1 - projected.y) * viewportHeight) / 2 - pixels / 2}px`;
    element.style.width = `${pixels}px`;
    element.style.height = `${pixels}px`;
    element.style.visibility =
      Math.abs(projected.x) > 1 ||
      Math.abs(projected.y) > 1 ||
      Math.abs(projected.z) > 1
        ? "hidden"
        : "visible";
  };
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const point = new THREE.Vector3(),
    previous = new THREE.Vector3(),
    color = new THREE.Color();
  const started = performance.now();
  let lastFrame = -Infinity;
  renderer.setAnimationLoop((now) => {
    // Bound CPU geometry/DOM work independently of display refresh rate.
    if (now - lastFrame < 1000 / 30) return;
    lastFrame = now;
    controls.update();
    const current = state();
    const values = stages.map(
      (stage) =>
        (current.trace ?? current.run)?.activations[stage.output]?.[0] ?? [],
    );
    nodes.forEach((node, index) => {
      const reached = current.trace
        ? current.progress >= node.layer
        : !!current.run;
      const value = values[node.layer][node.index] ?? 0;
      const selected =
        current.selectionKind === "neuron" &&
        current.selected === stages[node.layer].id &&
        current.neuron === node.index;
      color.setHex(
        selected
          ? 0xf1f5ff
          : !reached || value === 0
            ? 0x252b3b
            : value < 0
              ? 0xd5a995
              : 0x8fa8d4,
      );
      const relevant = relevantLayer(node.layer);
      neurons.setColorAt(index, color);
      const scale = relevant ? (selected ? 1.4 : 1) : 0;
      matrix.makeScale(scale, scale, scale).setPosition(node.position);
      neurons.setMatrixAt(index, matrix);
      if (relevant)
        syncTarget(
          node.position,
          selected ? 0.67 : 0.48,
          `${node.layer}:${node.index}`,
        );
      else {
        const target = targets.get(`${node.layer}:${node.index}`);
        if (target) target.style.visibility = "hidden";
      }
    });
    neurons.instanceMatrix.needsUpdate = true;
    if (neurons.instanceColor) neurons.instanceColor.needsUpdate = true;
    const time = reducedMotion.matches ? 0 : (now - started) / 1000;
    let particleCount = 0;
    edges.forEach((edge, index) => {
      const curve = edge.curve;
      const sway = Math.sin(time * 0.65 + edge.phase) * 0.14;
      const depth = Math.cos(time * 0.48 + edge.phase) * 0.22;
      curve.v1.lerpVectors(curve.v0, curve.v3, 0.33);
      curve.v1.y += -0.45 + sway;
      curve.v1.z += 0.3 + depth;
      curve.v2.lerpVectors(curve.v0, curve.v3, 0.67);
      curve.v2.y += -0.3 - sway * 0.6;
      curve.v2.z += -0.2 + depth;
      const contribution =
        (values[edge.layer - 1][edge.col] ?? 0) * edge.weight;
      const reached = current.trace
        ? current.progress >= edge.layer - 1
        : !!current.run;
      const picked =
        current.selectionKind === "edge" &&
        current.selection.tensorId === edge.tensorId &&
        current.selection.index[0] === edge.row &&
        current.selection.index[1] === edge.col;
      color.setHex(picked ? 0xffffff : contribution < 0 ? 0xd5a995 : 0x9aafd9);
      color.multiplyScalar(
        picked
          ? 1.6
          : !reached || contribution === 0
            ? 0.12
            : detail === "layer"
              ? 0.32
              : 0.6,
      );
      previous.copy(curve.v0);
      for (let segment = 0; segment < segments; segment++) {
        curve.getPoint((segment + 1) / segments, point);
        const offset = (index * segments + segment) * 6;
        positions[offset] = previous.x;
        positions[offset + 1] = previous.y;
        positions[offset + 2] = previous.z;
        positions[offset + 3] = point.x;
        positions[offset + 4] = point.y;
        positions[offset + 5] = point.z;
        for (let vertex = 0; vertex < 2; vertex++) {
          colors[offset + vertex * 3] = color.r;
          colors[offset + vertex * 3 + 1] = color.g;
          colors[offset + vertex * 3 + 2] = color.b;
        }
        previous.copy(point);
      }
      if (
        current.trace &&
        reached &&
        contribution !== 0 &&
        current.progress < edge.layer
      ) {
        curve.getPoint(current.progress - edge.layer + 1, point);
        particles.setMatrixAt(particleCount++, matrix.makeTranslation(point));
      }
      curve.getPoint(0.46, point);
      syncTarget(point, 0.18, edge.key);
    });
    positionAttribute.needsUpdate = true;
    colorAttribute.needsUpdate = true;
    // Raycasting requires a bound covering every animated segment.
    edgeGeometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3((stages.length - 1) * 3.5, 0, 0),
      stages.length * 7,
    );
    particles.count = particleCount;
    particles.instanceMatrix.needsUpdate = true;
    renderer.render(scene, camera);
    mount.dataset.drawCalls = String(renderer.info.render.calls);
    mount.dataset.renderedConnections = String(edges.length);
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
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
