import jax.numpy as jnp
from jax.numpy.linalg import inv
###############################################################################
# Define constants
###############################################################################
 
c      = 2.998e8       # m/s
GHz    = 1e9         # gigaHertz
THz    = 1e12        # teraHertz
deg    = jnp.pi/180.     # converts degrees to radians.
cm     = 0.01
mm     = 0.001
micron = 1e-6
inch   = 0.0254      # inch to meter conversion.
mil    = 2.54e-5     # mil (0.001") to meter conversion.

###############################################################################
# Start with definitions of objects that will be used in the code.
###############################################################################

class material(object):
    """
    Represents the properties of a given material,
    possibly a unixial crystal.
    """

    def __init__(self, ordinaryIndex, extraIndex,
                 ordinaryLoss, extraLoss, name, materialType='isotropic'):
        """
        Creates an instance of a material class object.
        Inputs are:

        ordinaryIndex - (float) real part of the refractive index for
                        the ordinary axes.
        extraIndex    - (float) real part of the refractive index for
                        the extraordinary axis. 
        ordinaryLoss  - (float) ratio of the imaginary part of the dielectric
                        constant to the real part for ordinary ray.
        extraLoss     - (float) ratio of the imaginary part of the dielectric
                        constant to the real part for extraordinary ray.
        name          - (string) name of the material in question.
        materialType  - (string) 'isotropic' or 'uniaxial'. If 'isotropic'
                        ordinaryIndex = extraIndex.
        """

        self.ordinaryIndex = ordinaryIndex
        self.extraIndex    = extraIndex
        self.extraLoss     = extraLoss
        self.ordinaryLoss  = ordinaryLoss
        self.name          = name
        self.materialType  = materialType

        # Now create complex dielectric constant and refractive indices.
        self.ordinaryEpsilon \
            = (1 - 1j*ordinaryLoss) * ordinaryIndex**2
        self.extraEpsilon \
            = (1 - 1j*extraLoss) * extraIndex**2

        self.ordinaryComplexIndex = jnp.sqrt(self.ordinaryEpsilon)
        self.extraComplexIndex    = jnp.sqrt(self.extraEpsilon)


    def __str__(self):
        """
        Print material properties in human-readable form.
        """

        s=[]
        s.append("\n")
        s.append("Material: %s"%(self.name))
        s.append("Material Type: %s"%(self.materialType))

        if self.materialType == 'isotropic':
            s.append("Refractive Index: %f"%(self.ordinaryIndex))
            s.append("Loss tangent: %f"%(self.ordinaryLoss))
        elif self.materialType == 'uniaxial':
            s.append("Refractive Index (ordinary): %f"%(self.ordinaryIndex))
            s.append("Refractive Index (extraordinary): %f"%(self.extraIndex))
            s.append("Loss Tangent (ordinary): %f"%(self.ordinaryLoss))
            s.append("Loss Tangent (extraordinary): %f"%(self.extraLoss))
        else: raise ValueError("""materialType is invalid\n.
                                Must be either 'isotropic' or 'uniaxial.'""")
        
        s.append("\n")

        return "\n".join(s)


class Stack(object):
    """
    An object to hold all the properties of a stack, including thicknesses,
    angles of any uniaxial crystals' extraordinary axes, and material objects
    for all the layer materials, as defined above.
    """

    def __init__(self, thicknesses, materials, angles):
        """
        Creates a Stack object using the following inputs:

        thicknesses  - list of thicknesses (float) in meters for the layers.
        materials    - list of material objects describing the materials of each
                       layer. The ordering is the same as the thicknesses list.
        angles       - important only for uniaxial crystal layers when there are
                       multiple uniaxial crystals at different angles in the stack.
                       Gives the angle in radians (float) between the extraordinary
                       axis and the x axis. Normally zero for all layers.
        """

        self.thicknesses = thicknesses
        self.materials   = materials
        self.angles      = angles

        self.numLayers = thicknesses.__len__()


    def __str__(self):
        """
        Prints the details of a stack in human-readable form
        """

        s = []
        s.append("\n")
        s.append("______________________________________________________")
        s.append("\n")

        for layerNum in range(self.numLayers):
            s.append("Layer %d: Thickness %f"%(layerNum+1, \
                                               self.thicknesses[layerNum]))
            materialString = self.materials[layerNum].__str__()
            s.append(materialString)
            if self.materials[layerNum].materialType == 'uniaxial':
                s.append("\t Angle of extraordinary axis to x-axis: %f"\
                         %(self.angles[layerNum]))
            s.append("______________________________________________________\n")

        return "\n".join(s)

        
class transferMatrix(object):         
    """
    Holds the transfer matrix for a single layer, along with
    the properties of the layer in a material object, defined above. This is
    calculated for a single frequency, index of refraction, and rotation of
    the layer about the z axis. This last rotation is only important for
    an anisotropic material.
    """

    def __init__(self, material, thickness, frequency, nsin, rotation):
        """
        Creates a transferMatrix object for a stack. Inputs are:

        material       - a material object that describes the properties of the layer.
        thickness      - thickness in meters of the layer.
        frequency      - frequency of incoming plane wave in Hz.
        nsin           - The invariant n*jnp.sin(theta) from Snell's Law.
        rotation       - angle that the stack is rotated about the z axis,
                         CCW from the x axis, also in radians.
        """
        
        self.material       = material
        self.frequency      = frequency
        self.nsin           = nsin
        self.rotation       = rotation

        self.optic_axis     = jnp.array((jnp.cos(rotation),jnp.sin(rotation),0),dtype=float)

        # Pull quantities that will show up in equations below.
        chi     = rotation
        t       = thickness
        k0      = 2*jnp.pi*frequency / c  # Wavenumber in free space.
        
        nO = material.ordinaryIndex
        nEmat = material.extraIndex
        nE = nEmat * jnp.sqrt(1 + (nEmat**(-2) - nO**(-2)) * nsin**2 *  jnp.cos(chi)**2)

        thetaO = jnp.arcsin( nsin / nO )
        thetaE = jnp.arcsin( nsin / nE )

        self.thetaO = thetaO
        self.thetaE = thetaE

        self.nE = nE        

        nComplexO = material.ordinaryComplexIndex
        nComplexE = jnp.sqrt(nE**2 *(1 - 1j*material.extraLoss))

        self.nComplexO = nComplexO
        self.nComplexE = nComplexE

        self.thetaOrdinary = thetaO
        self.thetaExtra    = thetaE
        self.extraordinaryIndex = nE
        self.ordinaryIndex = nO
        self.k0 = k0
        self.chi = chi
        self.ordinaryAngle = jnp.arcsin( nsin/nO )
        self.extraAngle    = jnp.arcsin( nsin/nE )

        ########################################################################
        # Calculate the rotated dielectric tensor for the material.
        ########################################################################

        rot    = jnp.array(((jnp.cos(rotation), -1*jnp.sin(rotation),0),\
                        (jnp.sin(rotation), jnp.cos(rotation), 0),\
                        (0, 0, 1)), dtype=complex)
        rotinv = inv(rot)
        eps    = jnp.array(((nE**2, 0, 0),(0, nO**2, 0),(0, 0, nO**2)), dtype=complex)
        roteps = jnp.dot( rot, jnp.dot( eps, rotinv))
        rotepsinv = inv(roteps)
        self.dielectric_tensor = roteps
        self.rot = rot
        self.rotinv = rotinv
        self.eps = eps
        
        ########################################################################
        # Calculate field components transmitted from Interface I.
        # All other fields can be related directly to these.
        ########################################################################

        # Electric displacement D = epsilon*E for ordinary ray.
        DenomDOrd = jnp.sqrt( jnp.cos(thetaO)**2 + jnp.sin(thetaO)**2 * jnp.sin(chi)**2)
        DOrd      = jnp.array(( -1*jnp.sin(chi)*jnp.cos(thetaO), \
                               jnp.cos(chi)*jnp.cos(thetaO), \
                               jnp.sin(chi)*jnp.sin(thetaO)), dtype=complex)

        DOrd      = DOrd / DenomDOrd
        self.DOrd = DOrd
        self.EOrd = jnp.dot(roteps, DOrd)


        # Electric displacement D = epsilon*E for extraordinary ray.
        DenomDExtra = jnp.sqrt( jnp.cos(chi)**2 *jnp.cos(thetaO)**2 + jnp.sin(chi)**2 * jnp.cos(thetaO-thetaE)**2)
        DExtra      = jnp.array(( 1*jnp.cos(chi)*jnp.cos(thetaO)*jnp.cos(thetaE), \
                                 1*jnp.sin(chi)*( jnp.sin(thetaO)*jnp.sin(thetaE) + jnp.cos(thetaO)*jnp.cos(thetaE)),\
                                 - jnp.cos(chi)*jnp.cos(thetaE)*jnp.sin(thetaO)), dtype=complex)
        DExtra      = DExtra / DenomDExtra
        self.DExtra = DExtra
        self.EExtra = jnp.dot(roteps, DExtra)

        # Magnetic field for ordinary ray.
        DenomHOrd = jnp.sqrt(jnp.cos(thetaO)**2 * jnp.cos(chi)**2 + jnp.sin(chi)**2)
        HOrd      = jnp.array(( -jnp.cos(thetaO)**2 * jnp.cos(chi),\
                               -jnp.sin(chi),\
                               jnp.cos(thetaO)*jnp.sin(thetaO)*jnp.cos(chi)), dtype=complex)
        HOrd      = HOrd / DenomHOrd
        self.HOrd = HOrd

        # Magnetic field for extraordinary ray.
        DenomHExtra = jnp.sqrt( jnp.cos(thetaO-thetaE)**2 * jnp.sin(chi)**2 + jnp.cos(thetaO)**2 * jnp.cos(chi)**2 ) 
        HExtra      = jnp.array(( -jnp.cos(thetaO-thetaE) * jnp.cos(thetaE) * jnp.sin(chi),\
                                 1*jnp.cos(thetaO)*jnp.cos(chi),\
                                 1*jnp.cos(thetaO-thetaE) * jnp.sin(thetaE) * jnp.sin(chi)),\
                               dtype=complex)
        HExtra      = HExtra / DenomHExtra
        self.HExtra = HExtra


        ##########################################################################
        # Form matrices that relate total fields at the two interfaces to the
        # field components above. The matrices will be used to eliminate the
        # field components above to get direct relations between the total fields
        # at the interfaces.
        ##########################################################################

        # Form the matrix relating vI = (Dx,Hy,Dy,-Hx) at interface I to the four
        # D components at the interface in the material. These are formed
        # into a vector v = (D^(o)_(tI), D^(e)_(tI), D^(o)_(r'II), D^(e)_(r'II)).
        # Then vI = M1.v

        Phi = jnp.array(((DOrd[0], DExtra[0], 1*DOrd[0], 1*DExtra[0]),\
                        (HOrd[1]/nO, HExtra[1]/nE, -1*HOrd[1]/nO, -1*HExtra[1]/nE),\
                        (DOrd[1], DExtra[1], DOrd[1], DExtra[1]),\
                        (-1*HOrd[0]/nO, -1*HExtra[0]/nE, HOrd[0]/nO, HExtra[0]/nE)),\
                      dtype=complex)

        # Form the matrix relating vII = (Dx,Hy,Dy,-Hx) at interface II to the four
        # D components at the interface in the material. These are formed
        # into a vector v = (D^(o)_(tI), D^(e)_(tI), D^(o)_(r'II), D^(e)_(r'II)).
        # Then vII = M1.v

        # The fields at the two interfaces are related adding a phase to the wave as it
        # travels across the medium. This phase depends on the index of refraction
        # and angle of travel, as well as the thickness of the material.

        deltaO = 1j * k0 * nComplexO * t * jnp.cos(thetaO)
        self.deltaO = deltaO
        deltaE = 1j * k0 * nComplexE * t * jnp.cos(thetaE)
        self.deltaE = deltaE

        # Define the propagation matrix P.
        P  = jnp.array( ((jnp.exp(-1*deltaO), 0, 0, 0),\
                        (0, jnp.exp(-1*deltaE), 0, 0),\
                        (0, 0, jnp.exp(deltaO), 0),\
                        (0, 0, 0, jnp.exp(deltaE))), dtype=complex)

        # Define the conversion matrix from D field components to those of E.
        Psi = jnp.array( ((rotepsinv[0,0], 0, rotepsinv[0,1], 0),\
                        (0, 1, 0, 0),\
                        (rotepsinv[1,0], 0, rotepsinv[1,1], 0),\
                        (0, 0, 0, 1)), dtype=complex)

        # Now compute the transfer matrix as Psi.Phi.inv(Psi.Phi.P)
        self.Phi = Phi
        self.P   = P
        self.Psi = Psi

        self.transferMatrix = jnp.dot( Psi, jnp.dot( Phi,jnp.linalg.inv(jnp.dot( Psi, jnp.dot(Phi, P)))))

        
class stackTransferMatrix(object):
    """
    Calculates the transfer matrix for a stack, and creates 
    an object which holds the individual layers plus the matrix for the stack as a whole.
    """

    def __init__(self, stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex):
        """
        Creates a stackTransferMatrix class instance, with inputs:

        stack          - Stack class instance holding information on the stack materials and thicknesses.
        frequency      - (float) frequency of incoming plane wave in Hz.
        incidenceAngle - (float) angle of incidence of the incoming plane wave in radians.
        rotation       - (float) rotation angle about the z-axis of the stack in radians.
        inputIndex     - (float) refractive index of medium containing incoming and reflected waves.
        exitIndex      - (float) refractive index of medium containing transmitted wave.
        """

        self.stack          = stack
        self.frequency      = frequency
        self.incidenceAngle = incidenceAngle
        self.rotation       = rotation
        self.inputIndex     = inputIndex
        self.exitIndex      = exitIndex
        self.nsin           = jnp.sin(incidenceAngle)*inputIndex

        numLayers   = stack.numLayers
        materials   = stack.materials
        thicknesses = stack.thicknesses
        angles      = stack.angles

        self.transfers = []
        self.totalTransfer = jnp.eye(4, dtype=complex)

        for layerNum in range(numLayers):
            material      = materials[layerNum]
            thickness     = thicknesses[layerNum]
            angle         = angles[layerNum]
            layerRotation = angle + rotation

            # Get the input and output angles for this layer. This depends on the
            # refractive indices of the materials before and after.
            if layerNum == 0: theta1 = incidenceAngle
            else: theta1 = jnp.arcsin( jnp.sin(incidenceAngle)*inputIndex / materials[layerNum-1].ordinaryIndex)

            if layerNum == (numLayers-1):
                theta3 = jnp.arcsin( jnp.sin(incidenceAngle)*inputIndex / exitIndex)
                self.exitAngle = theta3
            else: theta3 = jnp.arcsin( jnp.sin(incidenceAngle)*inputIndex / materials[layerNum+1].ordinaryIndex)

            layerTransfer = transferMatrix(material, thickness, frequency, self.nsin, layerRotation)
            self.transfers.append( layerTransfer )
            self.totalTransfer = jnp.dot( self.totalTransfer, layerTransfer.transferMatrix )


#################################################################################################################
# The functions below calculate Jones and Mueller matrices given a transfer matrix.
#################################################################################################################

def TranToJones(transfer):
    """
    Calculates the Jones matrices for reflected and transmitted waves from a transfer
    matrix. The input is a stackTransferMatrix class instance. Outputs a list where the first element is the
    2 x 2 Jones matrix for the transmitted wave and the second element is the 2 x 2 Jones matrix for
    the reflected wave.

    output - (list) list[0] is the transmitted jones matrix. list[1] is the reflected jones matrix.
    """

    # Grab the information we need from the stackTransferMatrix object.
    m      = transfer.totalTransfer # total transfer matrix.
    n1     = transfer.inputIndex
    n3     = transfer.exitIndex
    theta1 = transfer.incidenceAngle
    theta3 = transfer.exitAngle
   
    # Define a series of constants which will then go to making the relevant equations.
    A = (m[0,0] * jnp.cos(theta3) + m[0,1]*n3) / jnp.cos(theta1)
    B = (m[0,2] + m[0,3]*n3 * jnp.cos(theta3)) / jnp.cos(theta1)
    C = (m[1,0] * jnp.cos(theta3) + m[1,1]*n3) / n1
    D = (m[1,2] + m[1,3]*n3 * jnp.cos(theta3)) / n1
    N = (m[2,0] * jnp.cos(theta3) + m[2,1]*n3) 
    K = (m[2,2] + m[2,3]*n3 * jnp.cos(theta3)) 
    P = (m[3,0] * jnp.cos(theta3) + m[3,1]*n3) / (n1*jnp.cos(theta1))
    S = (m[3,2] + m[3,3]*n3 * jnp.cos(theta3)) / (n1*jnp.cos(theta1))

    # Now construct the Jones matrices for the reflected and transmitted rays.
    denom = (A+C)*(K+S)-(B+D)*(N+P) # Both matrices share a factor dividing all elements.

    # Transmitted Jones matrix.
    Jtran = jnp.array(((K+S, -B-D),(-N-P, A+C)), dtype=complex) * 2 / denom

    # Reflected Jones matrix.
    Jref = jnp.array(( ((C-A)*(K+S)-(D-B)*(N+P), 2*(A*D - C*B)),\
                      (2*(N*S - P*K), (A+C)*(K-S) - (D+B)*(N-P))),\
                    dtype=complex) / denom

    return Jtran, Jref


def JonesToMueller(jones):
    """
    Given a Jones matrix, computes the corresponding Mueller matrix. The input
    matrix should be a 2x2 complex jnp.array object.
    """

    # From the Pauli matrices into a list where Sigma[i] is the corresponding 2x2 matrix.
    # Note that they're in an unorthodox order, so that they match up properly with the
    # Stokes parameters. Note that to match up with Polarized Light by Goldstein,
    # I had to multiply the last Pauli matrix by -1, a change from the Jones et al paper.
    Sigma = []
    Sigma.append( jnp.array(( (1,0),(0,1)), dtype=complex)) # identity matrix
    Sigma.append( jnp.array(( (1,0),(0,-1)), dtype=complex))
    Sigma.append( jnp.array(( (0,1),(1,0)), dtype=complex))
    Sigma.append( jnp.array(( (0,-1j),(1j,0)), dtype=complex)) # Need to multiply by -1 to change back to normal.

    # Now the Mueller matrix elements are given by Mij = 1/2 * Tr(sigma[i]*J*sigma[j]*J^dagger)
    m = jnp.zeros((4,4), dtype=float)

    for i in range(4):
        for j in range(4):
            temp = jnp.trace( jnp.dot(Sigma[i], jnp.dot(jones, jnp.dot(Sigma[j], jones.conj().transpose())))) / 2

            # This is just a sanity check to make sure that the formula works and doesn't leave
            # an imaginary part floating around where the Mueller-matrix elements should be real.
            #if jnp.imag(temp) > 0.000000001: print('Discarding an imaginary part unnecessarily!!!!')
            m = m.at[i,j].set(jnp.real(temp))
    return m



############################################################################################################
# Scripts to go straight to Mueller and Jones matrices from a Stack object.
#
############################################################################################################

def Mueller(stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False):
    """
    Returns the Mueller matrix as a 4x4 numpy array for the given Stack, frequency, 
    angle of incidence, rotation of the stack, and optionally input and exit indices of refraction. 
    If you want the reflected Mueller matrix, set reflected to True.
    """

    transfer = stackTransferMatrix( stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex )
    jones   = TranToJones(transfer)

    if reflected==False:
        mueller = JonesToMueller( jones[0] )
    elif reflected==True:
        mueller = JonesToMueller( jones[1] )
    else: raise ValueError("""Invalid value for reflected. Must be True or False.""")

    return mueller




def Jones(stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False):
    """
    Returns the Jones matrix as a 2x2 complex numpy array for the given Stack, frequency,
    angle of incidence, rotation of the stack, and optionally input and exit indices of refraction.
    If you want the reflected Jones matrix, set reflected to True.
    """
    

    transfer = stackTransferMatrix( stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex )
    jones   = TranToJones(transfer)

    if reflected==False:
        output = jones[0] 
    elif reflected==True:
        output = jones[1]
    else: raise ValueError("""Invalid value for reflected. Must be True or False.""")

    return output
