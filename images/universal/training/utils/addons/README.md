# Jupyter Addons

This package contains addons for JupyterLab workbenches.

## Features / Bugs solved here

(second level bullet points indicate features/bugs that appeared due to the first level bullet point solution)

* [RHOAIENG-11156](https://issues.redhat.com/browse/RHOAIENG-11156) - Better feedback for JupyterLab-based workbenches initial load (improve time to first contentful paint)
  * [RHOAIENG-20553](https://issues.redhat.com/browse/RHOAIENG-20553) - CSS is broken when loading the TensorBoard extension

## Usage

The project uses PurgeCSS to tree-shake the PatternFly CSS file, removing unused styles.
The bundled output is generated in the `dist/` directory.

Code generation (generated code in `dist/` is committed to the repository)

```bash
pnpm install
pnpm build
```

Image build (in a Dockerfile)

```Dockerfile
ARG JUPYTER_REUSABLE_UTILS=jupyter/utils
WORKDIR /opt/app-root/bin
COPY ${JUPYTER_REUSABLE_UTILS} utils/
RUN # Apply JupyterLab addons \
    /opt/app-root/bin/utils/addons/apply.sh
```

## Development

### Example

Interactive demo of the spinner functionality:

1. Build the project: `pnpm build` or `pnpm build:dev`
2. Open `dist/index.html` in a browser
3. Clicking button simulates JupyterLab finished loading (spinner disappears)

### Build Process

The project uses webpack to bundle the JavaScript files and tree-shake the CSS:

- `pnpm build`: Creates a production build with minification
- `pnpm build:dev`: Creates a development build with source maps
- `pnpm build:clean`: Cleans the output directory and cache before building
- `pnpm clean`: Removes the dist directory and build cache
- `pnpm start`: Starts the webpack development server and opens `dist/index.html` (test page) in a browser
- `pnpm watch`: Watches for file changes and rebuilds automatically
- `pnpm test`: Runs the test-build.sh script to report tree-shaking effectiveness

## Files

- `apply.sh`: Script to apply the addons to a JupyterLab during Dockerfile build
- `partial-head.html`, `partial-body.html`: HTML content to be injected into the head section of JupyterLab
- `cleanup-webpack-plugin.mts`: Custom webpack plugin for asset cleanup (removes unnecessary files)
- `webpack.config.ts`: Webpack configuration with enhanced tree-shaking
- `dist/pf.css`: Tree-shaken PatternFly CSS file with only the necessary styles
- `src/index.ejs`: Template for the example page built into `dist/index.html`
- `dist/index.html`: Example HTML file demonstrating usage of the output
- `test-build.sh`: Script to verify the tree-shaking effectiveness
