import { useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

import { BAKE_FRAG, BAKE_VERT, MAX_CLUSTERS } from './planetShaders';
import { PLANET_SURFACE, DOMAIN_TERRAIN_TINT } from './palette';
import type { TileManifest } from '../tile/loader';

/** Equirectangular bake size by device tier. 2:1 aspect is mandatory. */
const SIZE_BY_TIER = { low: 1024, mid: 1536, high: 2048 } as const;

/**
 * Bakes the planet surface into a texture, once.
 *
 * Runs a fullscreen pass through an orthographic camera into a render target,
 * then throws the scene away. The result is a plain texture the globe samples
 * for the rest of the session — roughly 40ms of one-time work in exchange for
 * terrain that costs one fetch per fragment forever after.
 */
export function usePlanetTexture(manifest: TileManifest | null, tier: keyof typeof SIZE_BY_TIER) {
  const gl = useThree((s) => s.gl);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    if (!manifest) return;

    const width = SIZE_BY_TIER[tier];
    const height = width / 2;

    const target = new THREE.WebGLRenderTarget(width, height, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      // No mipmaps: the equirectangular seam at phi = 0 would bleed the far
      // edge of the map into the near one at every mip level, producing a
      // visible vertical scar down the globe.
      generateMipmaps: false,
    });
    // Wrapping matters at the seam even without mips — the last texel column
    // must blend into the first.
    target.texture.wrapS = THREE.RepeatWrapping;
    target.texture.wrapT = THREE.ClampToEdgeWrapping;
    // Linear, NOT sRGB. Every shader in this project is a raw ShaderMaterial
    // with no `colorspace_fragment` include, so nothing ever encodes to sRGB on
    // write. Tagging the target sRGB would make three decode on read without a
    // matching encode, and the whole planet would come out visibly darker.
    target.texture.colorSpace = THREE.LinearSRGBColorSpace;

    // Cluster centres become continental anchors. Padding entries are the zero
    // vector, which contributes nothing — the shader multiplies by length(mu).
    const clusters = new Array(MAX_CLUSTERS).fill(null).map(() => new THREE.Vector4(0, 0, 0, 0));
    manifest.clusters.slice(0, MAX_CLUSTERS).forEach((c, i) => {
      const st = Math.sin(c.theta);
      clusters[i].set(st * Math.cos(c.phi), Math.cos(c.theta), st * Math.sin(c.phi), c.domain);
    });

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uClusters: { value: clusters },
        uDomainTint: { value: DOMAIN_TERRAIN_TINT.map((c) => new THREE.Color(...c)) },
        uSeaLevel: { value: 0.0 },
        // Derived from the world seed so the map is reproducible: the same
        // world always grows the same continents.
        uSeed: { value: (manifest.seed % 97) * 0.37 },
        uDeepOcean: { value: new THREE.Color(...PLANET_SURFACE.deepOcean) },
        uShelf: { value: new THREE.Color(...PLANET_SURFACE.shelf) },
        uCoast: { value: new THREE.Color(...PLANET_SURFACE.coast) },
        uLowland: { value: new THREE.Color(...PLANET_SURFACE.lowland) },
        uHighland: { value: new THREE.Color(...PLANET_SURFACE.highland) },
        uIce: { value: new THREE.Color(...PLANET_SURFACE.ice) },
      },
      vertexShader: BAKE_VERT,
      fragmentShader: BAKE_FRAG,
      depthTest: false,
      depthWrite: false,
    });

    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    const scene = new THREE.Scene();
    scene.add(quad);
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const prevTarget = gl.getRenderTarget();
    gl.setRenderTarget(target);
    gl.render(scene, camera);
    gl.setRenderTarget(prevTarget);

    quad.geometry.dispose();
    material.dispose();

    setTexture(target.texture);
    return () => {
      target.dispose();
      setTexture(null);
    };
  }, [manifest, tier, gl]);

  return texture;
}
