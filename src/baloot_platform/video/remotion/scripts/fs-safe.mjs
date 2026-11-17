import {writeFile} from "node:fs/promises";

const retryableCodes = new Set(["EBUSY", "EACCES", "EPERM", "UNKNOWN"]);

export const writeFileWithRetry = async (file, data, options) => {
  let lastError;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      await writeFile(file, data, options);
      return;
    } catch (error) {
      lastError = error;
      if (!retryableCodes.has(error?.code) || attempt === 7) throw error;
      await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1)));
    }
  }
  throw lastError;
};
