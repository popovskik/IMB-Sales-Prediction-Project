/** "$1.5k"-style axis ticks; whole thousands drop the decimal ("$2k"). */
export const kTick = (v: number) => `$${(v / 1000).toFixed(1).replace(/\.0$/, "")}k`;
