# -*- coding: utf-8 -*-
"""
@author: Файл синтеза различных полей
"""
import numpy as np
from numpy import pi
import fields_synthesis as fs
import sys
sys.path.append('c:/MatkivskiyV/!Актуальная наука/Статья 7/python/')

#Функция для построения поля системой хручталик-объектив

class Field2D:
    
    def __init__(self,E,Nx=1024,Ny=1024,dx=.53,lam=.85):        
        
        self.lam  = lam
        self.Nx = Nx
        self.Ny = Ny        
        self.dx = dx        
        
        Lx = dx*Nx
        self.Lx  = Lx
        Ly = dx*Ny        
        self.Ly  = Ly
        k = 2*pi/lam
        self.k  = k
        
        xt = np.arange(-Lx/2, Lx/2, dx)
        yt = np.arange(-Ly/2, Ly/2, dx)
        
        x, y = np.meshgrid(xt, yt)
        
        knx = np.arange(-pi/dx, pi/dx, 2*pi/Lx)
        kny = np.arange(-pi/dx, pi/dx, 2*pi/Ly)
        
        kx, ky = np.meshgrid(knx, kny)        
        self.kz = np.sqrt(k**2-kx**2-ky**2)
    
    def pl(self,zl):
        self.E = fs.iFTS( fs.FTS(self.E)*np.exp(1j*self.kz*zl) )
        
    def mltp(self,E1):
        self.E *= E1
   
        

def pl(dz, kz, T):
    """
    dz - расстояние между плоскостями
    kz - фазовая маска
    T - поле, с которым производится операция """
    
    f = ( np.fft.ifftshift( np.fft.ifft2( np.fft.fft2(np.fft.fftshift(T))*
    np.fft.fftshift(np.exp(1j*kz*dz)) ) ) )        
    return f
def plh(dz, kz, r, T, lam, dx):
    """
    frensel impulse response propagation
    
    kz - фазовая маска
    T - поле, с которым производится операция """
    
    if (lam*dz/(T.shape[0]*dx)) < 1 :
        f = fs.iFTS( fs.FTS(T)*(np.exp(1j*kz*dz)) )
        
    else:
        k = 2*np.pi/lam
        fh = 1/(1j*lam*dz)*np.exp(1j*k*(r**2)/2/dz)
        fh = fs.FTS(fh)*dx**2 #нормировка
        f = fs.iFTS( fs.FTS(T)*fh )        
    return f
    
def circ(r,a):
    return r<=a
    
def rphase(dx,L,k):
    kn = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
    kx, ky = np.meshgrid(kn, kn)
    kz = 2*np.pi*np.sqrt((k/(2*np.pi))**2-kx**2-ky**2)
    return kz
    
def gauss_beam(sig, A, x, y):
    # Формируем гауссов пучок подсветки     
#    sig=1.5 #Диаметр пучка
    A = 100 #Амплитуда пучка
    S1 = ( (A/(sig*np.sqrt(2*np.pi)))*
    ( np.exp( -1*( x**2+y**2 )/(2*sig**2) ) ) )
    
    return S1

def gauss_beam2(sig, A, x, y):
    # Формируем гауссов пучок подсветки     
    # sig - Диаметр пучка
    # A - Амплитуда пучка
    S1 =  np.exp( -1*( x**2+y**2 )/(2*sig**2) )  
    
    return S1

def M(x, y, k, focus):
    r2 = (x**2+y**2)
    return  k*r2/(2*focus)
   
def FTS(T):
    return np.fft.fftshift( np.fft.fft2(np.fft.fftshift(T))  )

def iFTS(T):
    return np.fft.ifftshift( np.fft.ifft2(np.fft.ifftshift(T))  )

def size_transform(T,L,dx):
    #Задаешь новый размер массива и старый размер пикселя.
    #Функция строит то же поле но уже с большим размером пикселя, соответству
    #ющим новому размеру. Вообще процедура корявая конечно.
    l2 = np.int(T.shape[0]/2)
    N2 = np.int(L/dx/2)    
    fT = fs.FTS(T)
    S = fT[l2-N2:l2+N2,l2-N2:l2+N2]*(l2/(2*N2))**2
    return fs.iFTS(S)

def size_transform2(T,N):
    #принимает массив на вход и увеличивает изображение в нем в N праз
    if N >=  1:
        ld2 = np.int(T.shape[0]/2)
        l2d2 = np.int(ld2/N)    
        fS = fs.FTS(T[ld2-l2d2:ld2+l2d2,ld2-l2d2:ld2+l2d2])
        S = np.zeros((2*ld2,2*ld2),dtype='complex128')
        S[ld2-l2d2:ld2+l2d2,ld2-l2d2:ld2+l2d2] = fS
        return fs.iFTS(S)*N**2
    else:
        ld2 = np.int(T.shape[0]/2)
        l2d2 = np.int(ld2/N)
        fS = np.zeros((2*l2d2,2*l2d2),dtype='complex128')
        fS[l2d2-ld2:l2d2+ld2,l2d2-ld2:l2d2+ld2] = T
        fS = fs.FTS(fS)
        S  = fS[l2d2-ld2:l2d2+ld2,l2d2-ld2:l2d2+ld2]        
        return fs.iFTS(S)*N**2
    

def pixel_transform(T,dx1,dx2):
    # Принимает массив с размером пикселя dx1 и делает resampling на dx2
    Nd2 = np.int(T.shape[0]/2) #полуразмер первого массива
    N2d2 = np.int(Nd2*dx1/dx2) #полуразмер второго. Если пиксель увеличиваеися,
                               #то размер уменьшается и наоборот
    if N2d2 > Nd2:
        S = np.zeros((N2d2*2,N2d2*2), dtype='complex128')        
        S[N2d2-Nd2:N2d2+Nd2,N2d2-Nd2:N2d2+Nd2] = fs.FTS(T)
        S = fs.iFTS(S)*(dx1/dx2)**2
    else:
        S = fs.FTS(T)[Nd2-N2d2:Nd2+N2d2,Nd2-N2d2:Nd2+N2d2]
        S = fs.iFTS(S)*(dx1/dx2)**2
    return S

def shann_entropy(E):
    E =  E[E != 0]
    I = np.abs(E)**2    
    norm_I = I.sum()
    I /= norm_I
        
    return -1*(I*np.log2(I)).sum()
    
    
    
    
    
    
    
    
    