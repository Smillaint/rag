import { beforeAll, afterEach, afterAll } from "vitest";
import { network } from "./network.js";

beforeAll(async () => {
  await network.enable();
});

afterEach(() => {
  network.resetHandlers();
});

afterAll(async () => {
  await network.disable();
});
