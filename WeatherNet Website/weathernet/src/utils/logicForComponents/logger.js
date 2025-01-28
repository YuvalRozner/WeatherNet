const logger = {
  log: (...args) => {
    if (process.env.NODE_ENV === "development") {
      console.log(...args);
    }
  },
  info: (...args) => {
    if (process.env.NODE_ENV === "development") {
      console.info(...args);
    }
  },
  warn: (...args) => {
    if (process.env.NODE_ENV === "development") {
      console.warn(...args);
    }
  },
  error: (...args) => {
    console.error(...args); // Always log errors, regardless of environment
  },
};

export default logger;
