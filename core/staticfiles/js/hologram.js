// Hologram 3D Face Rendering with Three.js
document.addEventListener("DOMContentLoaded", function () {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true });

    const container = document.getElementById("hologram-container");
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const light = new THREE.PointLight(0x00ffff, 2, 50);
    light.position.set(2, 2, 5);
    scene.add(light);

    const loader = new THREE.GLTFLoader();
    loader.load("{% static 'models/face.glb' %}", function (gltf) {
        const model = gltf.scene;
        model.scale.set(1.5, 1.5, 1.5);
        scene.add(model);

        const material = new THREE.MeshStandardMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.6,
            emissive: 0x00ffff,
            wireframe: true,
        });

        model.traverse((child) => {
            if (child.isMesh) child.material = material;
        });

        let angle = 0;
        function animate() {
            requestAnimationFrame(animate);
            angle += 0.005;
            model.rotation.y = angle;
            renderer.render(scene, camera);
        }
        animate();
    });

    camera.position.z = 5;
});
