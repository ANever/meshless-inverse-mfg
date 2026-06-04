import subprocess
import os

def make_single_plot(doc, data_filename, ylabel=''):
    doc.write(r'''\begin{tikzpicture}
  \begin{axis}[%axis lines = center, %axis equal,
        grid,
        every axis y label/.style={
            at={(ticklabel cs:0.5,15pt)},rotate=90,anchor=center,
        }, 
        ymode = log,
        xmode = log,
        cycle list name= black white,
      x tick label style={rotate=45, anchor=east},
      legend style={
       at={(1.03,0.5)},
        anchor=west,
            % legend pos=north east,
       legend style={fill=none},
       },
      % legend to name=named,
      ylabel= Relative error,
      xlabel= Number of measurements,
    ]
    
    \def\file{data-errors''' + data_filename +r'''}
    
    \addplot+                  table [x=x, y=0, col sep=comma] {\file};
    \addplot+ [color=black!90] table [x=x, y=0.01, col sep=comma] {\file};
    \addplot+ [color=black!80] table [x=x, y=0.05, col sep=comma] {\file};
    \addplot+ [color=black!70] table [x=x, y=0.1, col sep=comma] {\file};
    \addplot+ [color=black!60] table [x=x, y=0.2, col sep=comma] {\file};
    
    \addlegendentry{$0$};
    \addlegendentry{$0.01$};
    \addlegendentry{$0.05$};
    \addlegendentry{$0.10$};
    \addlegendentry{$0.20$};
    
    
  \end{axis}
\end{tikzpicture}''')

def make_header(doc):
    doc.write(r'''
\documentclass[varwidth=2000pt, border=10pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{every axis plot post/.append style =
{samples=80, thick} } %mark options={fill=black}
\pgfplotsset{compat=1.18}
\pgfkeys{/pgf/number format/.cd,1000 sep={}}

\usepackage{amsmath}

\begin{document}
\centering
''')

def make_ending(doc):
    doc.write(r'''%\pgfplotslegendfromname{named}
    \end{document}''')    


#ylabels = ['$I_s^T$','$I_\gamma^T$','$E_s^T+E_\gamma^T$']


import pandas as pd
import subprocess
tex_filename = 'basic.tex'
#
for data_filename_prefix in  ['-straight/', '-straight-last/', '-pinn/', '-colloc/']:

    data_filename = 'rel_means.csv'

    data_filename = data_filename_prefix + data_filename
    if __name__ == "__main__":
        #pd.read_csv('data-errors-straight/'+data_filename, sep=',')
        with open('tex/' + tex_filename, 'w') as doc:
            make_header(doc)
            make_single_plot(doc, data_filename, ylabel='')
            doc.write(r'''\\ ''')
            make_ending(doc)
        subprocess.run(r"pdflatex --output-directory=results tex/"+ tex_filename, shell=True)
        subprocess.run(r"rename results\\basic.pdf errors" + data_filename[-8:-3] +"pdf", shell=True)


    data_filename = 'rel_means.csv'

    data_filename = data_filename_prefix + data_filename
    if __name__ == "__main__":
        #pd.read_csv('data-errors-straight/'+data_filename, sep=',')
        with open('tex/' + tex_filename, 'w') as doc:
            make_header(doc)
            make_single_plot(doc, data_filename, ylabel='')
            doc.write(r'''\\ ''')
            make_ending(doc)
        subprocess.run(r"pdflatex --output-directory=results tex/"+ tex_filename, shell=True)
        res_name = data_filename_prefix[:-1]+data_filename[-9:-3]
        subprocess.run(r"rename results\\basic.pdf errors" + res_name +"pdf", shell=True)