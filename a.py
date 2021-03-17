import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    
    x=np.arange(0.01,1,0.01)
    y=0.5*np.log((1-x)/x)
   
    plt.grid()    
    plt.subplots_adjust(top=0.9)
    plt.scatter(x,y,label=r'$\alpha =\frac{1}{2}\ln(\frac{1-\varepsilon}{\varepsilon })$')
    
    plt.legend()
    plt.xlabel(r'$\varepsilon$',fontsize=20)
    plt.ylabel(r'$\alpha$',fontsize=20)
    plt.xlim(0,1)    

    plt.show()