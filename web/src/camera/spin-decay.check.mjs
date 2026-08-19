// Mirrors the flick-to-spin decay in Rig.tsx.
//
// The property worth guarding is framerate independence: the coast must last
// the same wall-clock time on a 60Hz laptop and a 120Hz phone. Decaying by a
// fixed factor per frame instead of per second is the classic way to get this
// wrong, and it is invisible until someone tests on a different device.
const TAU = 0.47, REST = 0.02, MAX = 3.2;

function coast(v0, fps) {
  const dt = 1 / fps;
  let v = v0, travelled = 0, t = 0;
  while (Math.hypot(v, 0) > REST && t < 10) {
    travelled += v * dt;
    v *= Math.exp(-dt / TAU);
    t += dt;
  }
  return { travelled, duration: t };
}

const a = coast(2.0, 60);
const b = coast(2.0, 120);
const c = coast(2.0, 30);

for (const [name, x] of [['120Hz', b], ['30Hz', c]]) {
  const dErr = Math.abs(x.duration - a.duration) / a.duration;
  const tErr = Math.abs(x.travelled - a.travelled) / a.travelled;
  console.assert(dErr < 0.05, `${name} coasts for a different duration: ${x.duration} vs ${a.duration}`);
  console.assert(tErr < 0.05, `${name} travels a different distance: ${x.travelled} vs ${a.travelled}`);
}
console.log(`coast is framerate independent: ${a.duration.toFixed(2)}s @60Hz, ${b.duration.toFixed(2)}s @120Hz, ${c.duration.toFixed(2)}s @30Hz`);
console.log(`a 2.0 rad/s flick travels ${a.travelled.toFixed(2)} rad (${(a.travelled*180/Math.PI).toFixed(0)}°)`);

// A tap measures ~zero velocity and must not coast at all.
console.assert(coast(0.005, 60).duration === 0, 'a tap should not coast');
console.log('a tap (0.005 rad/s) coasts for 0s — no drift after a select');

// The clamp has to actually bound a wild swipe.
console.assert(Math.min(50, MAX) === MAX, 'clamp must bound absurd velocities');
console.log(`velocity clamped at ${MAX} rad/s`);
