import os
import io
import base64
import functools
import matplotlib.pyplot as plt

import numpy as np

__all__ = [
    'contract_objectives',
    'create_figure',
]

def contract_objectives(F,c,t):
    P = F[-1]
    stats = {
        'expected_pay': lambda t: P@t,
        'required_budget': lambda t: t.max(),
        'std_deviation': lambda t: np.sqrt(P@(t**2)-(P@t)**2)
    }
    return {k: f(t) for k,f in stats.items()}


background_line_style = {
    'color': 'lightgray',
    'linestyle': ':',
    'zorder': -100,
}

# Figures

class DownloadableIO(io.BytesIO):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._download_filename = filename

    def _repr_html_(self):
        buf = self.getbuffer()
        buf_enc = base64.b64encode(buf).decode('ascii')
        return f'<a href="data:text/plain;base64,{buf_enc}" download="{self._download_filename}">Download {self._download_filename}</a>'

def save_fig(fig, fname, **savefig_kwargs):
    return fig.savefig(
        fname=fname,
        # bbox_inches='tight',
        # pad_inches=0,
         **savefig_kwargs,
    )

def download_fig(fig, fname, **savefig_kwargs):
    fig_out = DownloadableIO(filename=os.path.basename(fname))
    save_fig(
        fig,
        fig_out,
        format=fname.split('.')[-1],
        **savefig_kwargs,
    )
    display(fig_out)

def save_and_download_fig(fig, fname, **savefig_kwargs):
    save_fig(fig, fname, **savefig_kwargs)
    print(f'Figure saved as {fname}')
    download_fig(fig, fname, **savefig_kwargs)

@functools.wraps(plt.subplots)
def create_figure(*args, **kwargs):
    kwargs['figsize'] = kwargs.pop('figsize',(10,3))
    kwargs['tight_layout'] = kwargs.pop('tight_layout',{'w_pad':2})
    fig, ax = plt.subplots(*args, **kwargs)
    fig.download = lambda filename: download_fig(fig, filename)
    fig.save_and_download = lambda filename: save_and_download_fig(fig, filename)
    return fig, ax

