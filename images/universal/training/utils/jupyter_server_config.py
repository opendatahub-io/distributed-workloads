# https://github.com/opendatahub-io-contrib/workbench-images/blob/main/snippets/ides/1-jupyter/files/etc/jupyter_notebook_config.py
c = get_config()
# Disable unsupported exporters
c.WebPDFExporter.enabled = False
c.QtPDFExporter.enabled = False
c.QtPNGExporter.enabled = False