// SPDX-License-Identifier: LicenseRef-Not-Copyrightable
import webpack from 'webpack';
import type { Compilation, Compiler, WebpackPluginInstance } from 'webpack';

interface CleanupPluginOptions {
    patterns: RegExp[];
}

class WebpackCleanupPlugin implements WebpackPluginInstance {
    private options: CleanupPluginOptions;

    constructor(options: CleanupPluginOptions) {
        this.options = {
            patterns: options.patterns || [],
        };
    }

    apply(compiler: Compiler) {
        compiler.hooks.thisCompilation.tap('WebpackCleanupPlugin', (compilation: Compilation) => {
            // Process assets to remove files based on filename patterns
            compilation.hooks.processAssets.tap(
                {
                    name: 'WebpackCleanupPlugin',
                    stage: webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONS,
                },
                (assets: Compilation['assets']) => {
                    // Remove files based on filename patterns
                    for (const filename in assets) {
                        if (this.options.patterns.some(pattern => pattern.test(filename))) {
                            compilation.deleteAsset(filename);
                        }
                    }
                },
            );
        });
    }
}

export default WebpackCleanupPlugin;
