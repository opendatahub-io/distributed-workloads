// https://webpack.js.org/configuration/configuration-languages/#typescript
import webpack from 'webpack';

import fs from 'node:fs';
import path from 'node:path';

import CleanupWebpackPlugin from './cleanup-webpack-plugin.ts';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import HtmlWebpackPlugin from "html-webpack-plugin";
import {PurgeCSSPlugin} from 'purgecss-webpack-plugin';

// Define __dirname for ES modules
// import { fileURLToPath } from 'node:url';
// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// Define the PatternFly CSS entry point
const patternflyCssEntry = './node_modules/@patternfly/patternfly/patternfly-no-globals.css';

// Helper function to read content of HTML files for PurgeCSS
const CONTENT_PATHS = [
  path.join(__dirname, 'partial-head.html'),
  path.join(__dirname, 'partial-body.html'),
];

// Create a cache directory if it doesn't exist
const CACHE_DIR = path.join(__dirname, '.cache');
if (!fs.existsSync(CACHE_DIR)) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
}

const config: webpack.Configuration = {
  mode: 'development',
  entry: {
    main: patternflyCssEntry
  },
  cache: {
    type: 'filesystem',
    cacheDirectory: CACHE_DIR,
  },
  // https://webpack.js.org/configuration/output/
  output: {
    path: path.resolve(__dirname, 'dist'),
    clean: true, // Clean the output directory before emit.
  },
  resolve: {
    extensions: ['.css']
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader,
          'css-loader'
        ]
      }
    ]
  },
  plugins: [
    // https://webpack.js.org/guides/output-management/#setting-up-htmlwebpackplugin
    new HtmlWebpackPlugin({
        title: 'Example spinner page',
        inject: false,
    }),
    // Use the cleanup plugin for any remaining files we want to remove
    new CleanupWebpackPlugin({
        patterns: [
            /^main.js$/, // useless empty js file
            /\.woff2$/, // Red Hat fonts used in pf.css
        ]
    }),
    // https://webpack.js.org/plugins/mini-css-extract-plugin/
    new MiniCssExtractPlugin({
      filename: 'pf.css',
    }),
    // https://purgecss.com/plugins/webpack.html#usage
    new PurgeCSSPlugin({
      paths: CONTENT_PATHS,
      safelist: {

        // NOTE: The following is discovered automatically by PurgeCSS
        standard: [],
        deep: [],

        // Preserve CSS variables that are used by the spinner
        // NOTE: The variables MUST be listed explicitly
        variables: [
          '--pf-t--global--icon',
          '--pf-t--global--animation',
          '--pf-v6-c-spinner',
          /--pf-v6-c-spinner--.*/,
          /--pf-v6-c-spinner__path--.*/,
        ]
      },

      // NOTE: this is also not necessary to specify
      blocklist: [],

      // Remove unused font faces and unnecessary styles
      fontFace: true,
      keyframes: true,
      variables: true,
    })
  ],
};

export default config;
