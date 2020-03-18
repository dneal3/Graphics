#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>

#include <cmath>
#include <algorithm>

#define NORMALS

using std::cerr;
using std::endl;

// ************************************************************************************************
// ************************** HANK'S SHADING CLASS *************************************************
// ************************************************************************************************

struct LightingParameters
{
    LightingParameters(void)
    {
        lightDir[0] = -0.6;
        lightDir[1] = 0;
        lightDir[2] = -0.8;
        Ka = 0.3;
        Kd = 0.7;
        Ks = 2.3;
        alpha = 2.5;
    };
    
    
    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
};

LightingParameters lp;

// ************************************************************************************************
// ************************** HANK'S MATRIX CLASS *************************************************
// ************************************************************************************************
class Matrix
{
public:
    double          A[4][4];
    
    void            TransformPoint(const double *ptIn, double *ptOut);
    static Matrix   ComposeMatrices(const Matrix &, const Matrix &);
    void            Print(ostream &o);
};

void
Matrix::Print(ostream &o)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        char str[256];
        sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
        o << str;
    }
}

Matrix
Matrix::ComposeMatrices(const Matrix &M1, const Matrix &M2)
{
    Matrix rv;
    for (int i = 0 ; i < 4 ; i++)
    for (int j = 0 ; j < 4 ; j++)
    {
        rv.A[i][j] = 0;
        for (int k = 0 ; k < 4 ; k++)
        rv.A[i][j] += M1.A[i][k]*M2.A[k][j];
    }
    
    return rv;
}

void
Matrix::TransformPoint(const double *ptIn, double *ptOut)
{
    ptOut[0] = ptIn[0]*A[0][0]
    + ptIn[1]*A[1][0]
    + ptIn[2]*A[2][0]
    + ptIn[3]*A[3][0];
    ptOut[1] = ptIn[0]*A[0][1]
    + ptIn[1]*A[1][1]
    + ptIn[2]*A[2][1]
    + ptIn[3]*A[3][1];
    ptOut[2] = ptIn[0]*A[0][2]
    + ptIn[1]*A[1][2]
    + ptIn[2]*A[2][2]
    + ptIn[3]*A[3][2];
    ptOut[3] = ptIn[0]*A[0][3]
    + ptIn[1]*A[1][3]
    + ptIn[2]*A[2][3]
    + ptIn[3]*A[3][3];
}
// ************************************************************************************************
// ************************************************************************************************
double dotProduct(double* v1, double* v2)
{
    double dot;
    dot = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
    return dot;
}

void divVector(double* v1, double v2)
{
    v1[0] = v1[0]/v2;
    v1[1] = v1[1]/v2;
    v1[2] = v1[2]/v2;
    v1[3] = v2;
}

double* subVectors(double* v1, double* v2)
{
    double* subd = new double[3];
    subd[0] = v1[0] - v2[0];
    subd[1] = v1[1] - v2[1];
    subd[2] = v1[2] - v2[2];
    return subd;
}

double* crossProduct(double *v1, double*v2)
{
    double* cross = new double[3];
    cross[0] = ((v1[1]*v2[2]) - (v1[2]*v2[1]));
    cross[1] = ((v2[0]*v1[2]) - (v1[0]*v2[2]));
    cross[2] = ((v1[0]*v2[1]) - (v1[1]*v2[0]));
    return cross;
}

void normalize(double* v1)
{
    double norm;
    norm = sqrt((v1[0]*v1[0]) + (v1[1]*v1[1]) + (v1[2]*v1[2]));
    v1[0] = v1[0]/norm;
    v1[1] = v1[1]/norm;
    v1[2] = v1[2]/norm;
}

// ************************************************************************************************
// ************************** HANK'S CAMERA CLASS *************************************************
// ************************************************************************************************
class Camera
{
public:
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];

    
    Matrix          ViewTransform(void);
    Matrix          CameraTransform(void);
    Matrix          DeviceTransform(void);
};

Matrix Camera::CameraTransform(void)
{
    double* origin = new double[3];
    origin[0] = origin[1] = origin[2] = 0;
    double* t = subVectors(origin, position);
    double* w = subVectors(position, focus);
    double* u = crossProduct(up, w);
    double* v = crossProduct(w, u);
    
    normalize(u);
    normalize(v);
    normalize(w);

    Matrix* ct = new Matrix();
    
    ct->A[0][0] = u[0];
    ct->A[0][1] = v[0];
    ct->A[0][2] = w[0];
    ct->A[0][3] = 0;
    
    ct->A[1][0] = u[1];
    ct->A[1][1] = v[1];
    ct->A[1][2] = w[1];
    ct->A[1][3] = 0;
    
    ct->A[2][0] = u[2];
    ct->A[2][1] = v[2];
    ct->A[2][2] = w[2];
    ct->A[2][3] = 0;
    
    ct->A[3][0] = dotProduct(u, t);
    ct->A[3][1] = dotProduct(v, t);
    ct->A[3][2] = dotProduct(w, t);
    ct->A[3][3] = 1;
    
    return *ct;
}

Matrix Camera::ViewTransform(void)
{
    Matrix* vt = new Matrix();
    
    vt->A[0][0] = 1/tan((angle/2));
    vt->A[0][1] = 0;
    vt->A[0][2] = 0;
    vt->A[0][3] = 0;
    
    vt->A[1][0] = 0;
    vt->A[1][1] = 1/tan((angle/2));
    vt->A[1][2] = 0;
    vt->A[1][3] = 0;
    
    vt->A[2][0] = 0;
    vt->A[2][1] = 0;
    vt->A[2][2] = (far+near)/(far-near);
    vt->A[2][3] = -1;
    
    vt->A[3][0] = 0;
    vt->A[3][1] = 0;
    vt->A[3][2] = (2*(far*near))/(far-near);
    vt->A[3][3] = 0;
    
    return *vt;
}

Matrix Camera::DeviceTransform(void)
{
    Matrix* dt = new Matrix();
    int n = 1000; // could change this so it isnt hardcoded
    int m = 1000;
    
    
    dt->A[0][0] = n/2;
    dt->A[0][1] = 0;
    dt->A[0][2] = 0;
    dt->A[0][3] = 0;
    
    dt->A[1][0] = 0;
    dt->A[1][1] = m/2;
    dt->A[1][2] = 0;
    dt->A[1][3] = 0;
    
    dt->A[2][0] = 0;
    dt->A[2][1] = 0;
    dt->A[2][2] = 1;
    dt->A[2][3] = 0;
    
    dt->A[3][0] = n/2;
    dt->A[3][1] = m/2;
    dt->A[3][2] = 0;
    dt->A[3][3] = 1;
    
    return *dt;
}




double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

Camera
GetCamera(int frame, int nframes)
{
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0;
    c.focus[1] = 0;
    c.focus[2] = 0;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}
// ************************************************************************************************
// ************************************************************************************************

double ceil441(double f)
{
    return ceil(f-0.00001);
}

double floor441(double f)
{
    return floor(f+0.00001);
}





vtkImageData *
NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

    return img;
}

void
WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}

class Edge
{
    public:
        double X[2];
        double Y[2];
        double Z[2];
        double shading[2];
        double colors[2][3];
        double slope;
        double yint;
        double currbound[2];
};

class EdgeTable
{
    public:
        Edge*  edges[3];
        double midy;
        double maxy;
        double miny;
    
};

class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      double         Z[3];
      double colors[3][3];
      double normals[3][3];
      double shading[3];
};

class Screen
{
  public:
      double *zbuffer;
      unsigned char   *buffer;
      int width, height;

};

Edge* AnalyzePoints(double x0, double y0, double x1, double y1)
{
    Edge* edge = new Edge();
    edge->X[0] = x0;
    edge->X[1] = x1;
    edge->Y[0] = y0;
    edge->Y[1] = y1;

    double rise = y1 - y0;
    double run = x1 - x0;
    
    if(x1 == x0)
    {
        edge->slope = NAN;
        edge->yint = 0;
    }
    else
    {
        edge->slope = (rise/run);
        edge->yint = ((edge->slope*(-1.0*x0)) + y0);
    }
    return edge;
}

double Interpolate(double A, double B, double FA, double FB, double X)
{
    double t;
    double FX;
    
    t = (X-A)/(B-A);
    
    FX = FA + t*(FB-FA);
    
    return FX;
}

double ComputeBoundary(double y, Edge* edge)
{

    if(isnan(edge->slope) || edge->slope == 0)
    {
        edge->currbound[0] = edge->X[0];
        edge->currbound[1] = y;
        return edge->currbound[0];
    }
    else
    {
        edge->currbound[0] = ((y - edge->yint)/(edge->slope));
        return edge->currbound[0];
    }
}

void ColorTri(EdgeTable* edgetable, unsigned char* buffer, Screen screen)
{
  
    double leftbound;
    double rightbound;
    double interpolatedshadingleft;
    double interpolatedshadingright;
    double interpolatedmiddleshading;
    double interpolatedleftz;
    double interpolatedrightz;
    double interpolatedmiddlez;
    double interpolatedleft[3];
    double interpolatedright[3];
    double maxy = edgetable->maxy;
    double miny = edgetable->miny;
    if(miny < 0)
        miny = 0;
    for(int i = ceil441(miny); i<=floor441(maxy); i++) // mid should equal max as they have same y value
    {
        if(i >= screen.height)
            break;
        
        int linestart = (i*screen.width)*3;
        leftbound = ComputeBoundary(i, edgetable->edges[0]);
        rightbound = ComputeBoundary(i, edgetable->edges[1]);

        //make the interpolated colors of the left bound to be used while coloring
        interpolatedleft[0] = Interpolate(edgetable->edges[0]->Y[0], edgetable->edges[0]->Y[1], edgetable->edges[0]->colors[0][0], edgetable->edges[0]->colors[1][0], i);
        interpolatedleft[1] = Interpolate(edgetable->edges[0]->Y[0], edgetable->edges[0]->Y[1], edgetable->edges[0]->colors[0][1], edgetable->edges[0]->colors[1][1], i);
        interpolatedleft[2] = Interpolate(edgetable->edges[0]->Y[0], edgetable->edges[0]->Y[1], edgetable->edges[0]->colors[0][2], edgetable->edges[0]->colors[1][2], i);
        interpolatedleftz = Interpolate(edgetable->edges[0]->Y[0], edgetable->edges[0]->Y[1], edgetable->edges[0]->Z[0], edgetable->edges[0]->Z[1], i);
        interpolatedshadingleft = Interpolate(edgetable->edges[0]->Y[0], edgetable->edges[0]->Y[1], edgetable->edges[0]->shading[0], edgetable->edges[0]->shading[1], i);
        
        //make the interpolated colors of the right bound to be used while coloring
        interpolatedright[0] = Interpolate(edgetable->edges[1]->Y[0], edgetable->edges[1]->Y[1], edgetable->edges[1]->colors[0][0], edgetable->edges[1]->colors[1][0], i);
        interpolatedright[1] = Interpolate(edgetable->edges[1]->Y[0], edgetable->edges[1]->Y[1], edgetable->edges[1]->colors[0][1], edgetable->edges[1]->colors[1][1], i);
        interpolatedright[2] = Interpolate(edgetable->edges[1]->Y[0], edgetable->edges[1]->Y[1], edgetable->edges[1]->colors[0][2], edgetable->edges[1]->colors[1][2], i);
        interpolatedrightz = Interpolate(edgetable->edges[1]->Y[0], edgetable->edges[1]->Y[1], edgetable->edges[1]->Z[0], edgetable->edges[1]->Z[1], i);
        interpolatedshadingright = Interpolate(edgetable->edges[1]->Y[0], edgetable->edges[1]->Y[1], edgetable->edges[1]->shading[0], edgetable->edges[1]->shading[1], i);
        
        for(int j = (ceil441(leftbound)); j <= (floor441(rightbound)); j++)
        {
            if(j >= screen.width) // not doing things outside bounds
                break;
            if(j < 0)
                j = 0;

            int pix = j*3 + linestart;
            
            interpolatedmiddlez = Interpolate(leftbound, rightbound, interpolatedleftz, interpolatedrightz, j);
            interpolatedmiddleshading = Interpolate(leftbound, rightbound, interpolatedshadingleft, interpolatedshadingright, j);
            //printf("overall shading value is %f\n", interpolatedmiddleshading);
            if(interpolatedmiddlez > screen.zbuffer[j+(i*screen.width)])
            {
                screen.zbuffer[j+(i*screen.width)] = interpolatedmiddlez;
                buffer[pix] =   ceil441(fmin((Interpolate(leftbound, rightbound, interpolatedleft[0], interpolatedright[0], j) * interpolatedmiddleshading * 255.0), 255.0));
                buffer[pix+1] = ceil441(fmin((Interpolate(leftbound, rightbound, interpolatedleft[1], interpolatedright[1], j) * interpolatedmiddleshading * 255.0), 255.0));
                buffer[pix+2] = ceil441(fmin((Interpolate(leftbound, rightbound, interpolatedleft[2], interpolatedright[2], j) * interpolatedmiddleshading * 255.0), 255.0));
            }
        }
    }
}

void AnalyzeEdges(EdgeTable* edgetable, unsigned char* buffer, Screen screen)
{
    int maxflag = 0;
    int minflag = 0;
    double midXYZ[3];
    midXYZ[0] = 0;
    double maxXYZ[3];
    maxXYZ[0] = 0;
    double minXYZ[3];
    minXYZ[0] = 0;
    double newXYZ[3];
    double maxColor[3];
    double midColor[3];
    double minColor[3];
    double newColor[3];
    double midShading;
    double minShading;
    double maxShading;
    double newShading;
    EdgeTable* uptritable = new EdgeTable();
    EdgeTable* downtritable = new EdgeTable();
    Edge* intersectedge = new Edge();
    
    //This will find the intersect edges which is necessary for analyzing arbitary triangles, but this will also find the ordering of edges and will set up an edgetable for a triangle so that assumptions could be made when trying to color a triangle.
    for(int i = 0; i<3; i++)
    {
        
        if((edgetable->edges[i]->Y[0] != edgetable->midy) && (edgetable->edges[i]->Y[1] != edgetable->midy))
        {
            intersectedge = edgetable->edges[i];
        }
        for(int j = 0; j < 2; j++)
        {
            if(midXYZ[0] && maxXYZ[0] && minXYZ[0]) // mild optimization
                break;
            //need a special condition to find a "mid point" for a going up or down triangle, these next two if statements do that.
            if((edgetable->edges[i]->Y[j] == edgetable->maxy) && (edgetable->edges[i]->X[j] != maxXYZ[0]) && maxflag)
            {
                midXYZ[0] = edgetable->edges[i]->X[j];
                midXYZ[1] = edgetable->edges[i]->Y[j];
                midXYZ[2] = edgetable->edges[i]->Z[j];
                midShading = edgetable->edges[i]->shading[j];
                midColor[0] = edgetable->edges[i]->colors[j][0];
                midColor[1] = edgetable->edges[i]->colors[j][1];
                midColor[2] = edgetable->edges[i]->colors[j][2];
            }
            else if((edgetable->edges[i]->Y[j] == edgetable->miny) && (edgetable->edges[i]->X[j] != minXYZ[0]) && minflag)
            {
                midXYZ[0] = edgetable->edges[i]->X[j];
                midXYZ[1] = edgetable->edges[i]->Y[j];
                midXYZ[2] = edgetable->edges[i]->Z[j];
                midShading = edgetable->edges[i]->shading[j];
                midColor[0] = edgetable->edges[i]->colors[j][0];
                midColor[1] = edgetable->edges[i]->colors[j][1];
                midColor[2] = edgetable->edges[i]->colors[j][2];
            }
            else if(edgetable->edges[i]->Y[j] == edgetable->maxy)
            {
                maxXYZ[0] = edgetable->edges[i]->X[j];
                maxXYZ[1] = edgetable->edges[i]->Y[j];
                maxXYZ[2] = edgetable->edges[i]->Z[j];
                maxShading = edgetable->edges[i]->shading[j];
                maxColor[0] = edgetable->edges[i]->colors[j][0];
                maxColor[1] = edgetable->edges[i]->colors[j][1];
                maxColor[2] = edgetable->edges[i]->colors[j][2];
                maxflag = 1;
            }
            else if(edgetable->edges[i]->Y[j] == edgetable->miny)
            {
                minXYZ[0] = edgetable->edges[i]->X[j];
                minXYZ[1] = edgetable->edges[i]->Y[j];
                minXYZ[2] = edgetable->edges[i]->Z[j];
                minShading = edgetable->edges[i]->shading[j];
                minColor[0] = edgetable->edges[i]->colors[j][0];
                minColor[1] = edgetable->edges[i]->colors[j][1];
                minColor[2] = edgetable->edges[i]->colors[j][2];
                minflag = 1;
            }
            else
            {
                //the midpoint for an arbitrary triangle
                midXYZ[0] = edgetable->edges[i]->X[j];
                midXYZ[1] = edgetable->edges[i]->Y[j];
                midXYZ[2] = edgetable->edges[i]->Z[j];
                midShading = edgetable->edges[i]->shading[j];
                midColor[0] = edgetable->edges[i]->colors[j][0];
                midColor[1] = edgetable->edges[i]->colors[j][1];
                midColor[2] = edgetable->edges[i]->colors[j][2];
            }
        }
    }
    
    //printf("minShading: %f\n", minShading);
    //printf("midShading: %f\n", midShading);
    //printf("maxShading: %f\n", maxShading);
    
    //printf("miny = %f\n", minXYZ[1]);
    //printf("midy = %f\n", midXYZ[1]);
    //printf("maxy = %f\n", maxXYZ[1]);
    
    double tempShading;
    double tempXYZ[3];
    double tempColors[3];
    
    if(minXYZ[1] == midXYZ[1])
    {
        //this is the code for generating a going up triangle and color it
        if(midXYZ[0] < minXYZ[0])
        {
            tempXYZ[0] = minXYZ[0];
            tempXYZ[1] = minXYZ[1];
            tempXYZ[2] = minXYZ[2];
            tempShading = minShading;
            minXYZ[0] = midXYZ[0];
            minXYZ[1] = midXYZ[1];
            minXYZ[2] = midXYZ[2];
            minShading = midShading;
            midXYZ[0] = tempXYZ[0];
            midXYZ[1] = tempXYZ[1];
            midXYZ[2] = tempXYZ[2];
            midShading = tempShading;
            
            tempColors[0] = minColor[0];
            tempColors[1] = minColor[1];
            tempColors[2] = minColor[2];
            minColor[0] = midColor[0];
            minColor[1] = midColor[1];
            minColor[2] = midColor[2];
            midColor[0] = tempColors[0];
            midColor[1] = tempColors[1];
            midColor[2] = tempColors[2];
        }
        
        uptritable->edges[0] = AnalyzePoints(minXYZ[0], minXYZ[1], maxXYZ[0], maxXYZ[1]); // left bound
        uptritable->edges[0]->Z[0] = minXYZ[2];
        uptritable->edges[0]->Z[1] = maxXYZ[2];
        uptritable->edges[0]->colors[0][0] = minColor[0];
        uptritable->edges[0]->colors[0][1] = minColor[1];
        uptritable->edges[0]->colors[0][2] = minColor[2];
        uptritable->edges[0]->colors[1][0] = maxColor[0];
        uptritable->edges[0]->colors[1][1] = maxColor[1];
        uptritable->edges[0]->colors[1][2] = maxColor[2];
        uptritable->edges[0]->shading[0] = minShading;
        uptritable->edges[0]->shading[1] = maxShading;
        
        uptritable->edges[1] = AnalyzePoints(midXYZ[0], midXYZ[1], maxXYZ[0], maxXYZ[1]); // right bound
        uptritable->edges[1]->Z[0] = midXYZ[2];
        uptritable->edges[1]->Z[1] = maxXYZ[2];
        uptritable->edges[1]->colors[0][0] = midColor[0];
        uptritable->edges[1]->colors[0][1] = midColor[1];
        uptritable->edges[1]->colors[0][2] = midColor[2];
        uptritable->edges[1]->colors[1][0] = maxColor[0];
        uptritable->edges[1]->colors[1][1] = maxColor[1];
        uptritable->edges[1]->colors[1][2] = maxColor[2];
        uptritable->edges[1]->shading[0] = midShading;
        uptritable->edges[1]->shading[1] = maxShading;
        
        uptritable->edges[2] = AnalyzePoints(minXYZ[0], minXYZ[1], midXYZ[0], midXYZ[1]); // new min edge
        uptritable->edges[2]->Z[0] = minXYZ[2];
        uptritable->edges[2]->Z[1] = midXYZ[2];
        uptritable->edges[2]->colors[0][0] = minColor[0];
        uptritable->edges[2]->colors[0][1] = minColor[1];
        uptritable->edges[2]->colors[0][2] = minColor[2];
        uptritable->edges[2]->colors[1][0] = midColor[0];
        uptritable->edges[2]->colors[1][1] = midColor[1];
        uptritable->edges[2]->colors[1][2] = midColor[2];
        uptritable->edges[2]->shading[0] = minShading;
        uptritable->edges[2]->shading[1] = midShading;
        
        uptritable->miny = minXYZ[1];
        uptritable->maxy = maxXYZ[1];
        uptritable->midy = midXYZ[1];
        ColorTri(uptritable, buffer, screen);
    }
    else if(maxXYZ[1] == midXYZ[1])
    {
        //this is the code for generating a going down triangle and coloring it.
        if(maxXYZ[0] > midXYZ[0])
        {
            tempXYZ[0] = maxXYZ[0];
            tempXYZ[1] = maxXYZ[1];
            tempXYZ[2] = maxXYZ[2];
            tempShading = maxShading;
            maxXYZ[0] = midXYZ[0];
            maxXYZ[1] = midXYZ[1];
            maxXYZ[2] = midXYZ[2];
            maxShading = midShading;
            midXYZ[0] = tempXYZ[0];
            midXYZ[1] = tempXYZ[1];
            midXYZ[2] = tempXYZ[2];
            midShading = tempShading;
            
            tempColors[0] = maxColor[0];
            tempColors[1] = maxColor[1];
            tempColors[2] = maxColor[2];
            maxColor[0] = midColor[0];
            maxColor[1] = midColor[1];
            maxColor[2] = midColor[2];
            midColor[0] = tempColors[0];
            midColor[1] = tempColors[1];
            midColor[2] = tempColors[2];
        }
        
        
        downtritable->edges[0] = AnalyzePoints(maxXYZ[0], maxXYZ[1], minXYZ[0], minXYZ[1]); // left bound
        downtritable->edges[0]->Z[0] = maxXYZ[2];
        downtritable->edges[0]->Z[1] = minXYZ[2];
        downtritable->edges[0]->colors[0][0] = maxColor[0];
        downtritable->edges[0]->colors[0][1] = maxColor[1];
        downtritable->edges[0]->colors[0][2] = maxColor[2];
        downtritable->edges[0]->colors[1][0] = minColor[0];
        downtritable->edges[0]->colors[1][1] = minColor[1];
        downtritable->edges[0]->colors[1][2] = minColor[2];
        downtritable->edges[0]->shading[0] = maxShading;
        downtritable->edges[0]->shading[1] = minShading;
        
        downtritable->edges[1] = AnalyzePoints(midXYZ[0], midXYZ[1], minXYZ[0], minXYZ[1]); // right bound
        downtritable->edges[1]->Z[0] = midXYZ[2];
        downtritable->edges[1]->Z[1] = minXYZ[2];
        downtritable->edges[1]->colors[0][0] = midColor[0];
        downtritable->edges[1]->colors[0][1] = midColor[1];
        downtritable->edges[1]->colors[0][2] = midColor[2];
        downtritable->edges[1]->colors[1][0] = minColor[0];
        downtritable->edges[1]->colors[1][1] = minColor[1];
        downtritable->edges[1]->colors[1][2] = minColor[2];
        downtritable->edges[1]->shading[0] = midShading;
        downtritable->edges[1]->shading[1] = minShading;
        
        downtritable->edges[2] = AnalyzePoints(maxXYZ[0], maxXYZ[1], midXYZ[0], midXYZ[1]);
        downtritable->edges[2]->Z[0] = maxXYZ[2];
        downtritable->edges[2]->Z[1] = midXYZ[2];
        downtritable->edges[2]->colors[0][0] = maxColor[0];
        downtritable->edges[2]->colors[0][1] = maxColor[1];
        downtritable->edges[2]->colors[0][2] = maxColor[2];
        downtritable->edges[2]->colors[1][0] = midColor[0];
        downtritable->edges[2]->colors[1][1] = midColor[1];
        downtritable->edges[2]->colors[1][2] = midColor[2];
        downtritable->edges[2]->shading[0] = maxShading;
        downtritable->edges[2]->shading[1] = midShading;
        
        
        downtritable->miny = minXYZ[1];
        downtritable->maxy = maxXYZ[1];
        downtritable->midy = midXYZ[1];
        ColorTri(downtritable, buffer, screen);
    }
    else
    {
        
        //this is the code for generating a going down and up triangle to represent an arbitrary triangle
        newXYZ[0] = ComputeBoundary(midXYZ[1], intersectedge);
        newXYZ[1] = midXYZ[1];
        newColor[0] = Interpolate(minXYZ[1], maxXYZ[1], minColor[0], maxColor[0], midXYZ[1]);
        newColor[1] = Interpolate(minXYZ[1], maxXYZ[1], minColor[1], maxColor[1], midXYZ[1]);
        newColor[2] = Interpolate(minXYZ[1], maxXYZ[1], minColor[2], maxColor[2], midXYZ[1]);
        newXYZ[2] = Interpolate(minXYZ[1], maxXYZ[1], minXYZ[2], maxXYZ[2], midXYZ[1]);
        //newShading = Interpolate(minXYZ[1], maxXYZ[1], minShading, maxShading, midXYZ[1]);
        newShading = Interpolate(intersectedge->Y[0], intersectedge->Y[1], intersectedge->shading[0], intersectedge->shading[1], midXYZ[1]);
        //printf("newShading %f\n", newShading);
        
        
        if(newXYZ[0] < midXYZ[0])
        {
            tempXYZ[0] = newXYZ[0];
            tempXYZ[1] = newXYZ[1];
            tempXYZ[2] = newXYZ[2];
            tempShading = newShading;
            newXYZ[0] = midXYZ[0];
            newXYZ[1] = midXYZ[1];
            newXYZ[2] = midXYZ[2];
            newShading = midShading;
            midXYZ[0] = tempXYZ[0];
            midXYZ[1] = tempXYZ[1];
            midXYZ[2] = tempXYZ[2];
            midShading = tempShading;
            
            tempColors[0] = newColor[0];
            tempColors[1] = newColor[1];
            tempColors[2] = newColor[2];
            newColor[0] = midColor[0];
            newColor[1] = midColor[1];
            newColor[2] = midColor[2];
            midColor[0] = tempColors[0];
            midColor[1] = tempColors[1];
            midColor[2] = tempColors[2];
        }
        
        
        uptritable->edges[0] = AnalyzePoints(maxXYZ[0], maxXYZ[1], midXYZ[0], midXYZ[1]); // left bound
        uptritable->edges[0]->Z[0] = maxXYZ[2];
        uptritable->edges[0]->Z[1] = midXYZ[2];
        uptritable->edges[0]->colors[0][0] = maxColor[0];
        uptritable->edges[0]->colors[0][1] = maxColor[1];
        uptritable->edges[0]->colors[0][2] = maxColor[2];
        uptritable->edges[0]->colors[1][0] = midColor[0];
        uptritable->edges[0]->colors[1][1] = midColor[1];
        uptritable->edges[0]->colors[1][2] = midColor[2];
        uptritable->edges[0]->shading[0] = maxShading;
        uptritable->edges[0]->shading[1] = midShading;
        
        uptritable->edges[1] = AnalyzePoints(maxXYZ[0], maxXYZ[1], newXYZ[0], newXYZ[1]); // right bound
        uptritable->edges[1]->Z[0] = maxXYZ[2];
        uptritable->edges[1]->Z[1] = newXYZ[2];
        uptritable->edges[1]->colors[0][0] = maxColor[0];
        uptritable->edges[1]->colors[0][1] = maxColor[1];
        uptritable->edges[1]->colors[0][2] = maxColor[2];
        uptritable->edges[1]->colors[1][0] = newColor[0];
        uptritable->edges[1]->colors[1][1] = newColor[1];
        uptritable->edges[1]->colors[1][2] = newColor[2];
        uptritable->edges[1]->shading[0] = maxShading;
        uptritable->edges[1]->shading[1] = newShading;
        
        uptritable->edges[2] = AnalyzePoints(midXYZ[0], midXYZ[1], newXYZ[0], newXYZ[1]);
        uptritable->edges[2]->Z[0] = midXYZ[2];
        uptritable->edges[2]->Z[1] = newXYZ[2];
        uptritable->edges[2]->colors[0][0] = midColor[0];
        uptritable->edges[2]->colors[0][1] = midColor[1];
        uptritable->edges[2]->colors[0][2] = midColor[2];
        uptritable->edges[2]->colors[1][0] = newColor[0];
        uptritable->edges[2]->colors[1][1] = newColor[1];
        uptritable->edges[2]->colors[1][2] = newColor[2];
        uptritable->edges[2]->shading[0] = midShading;
        uptritable->edges[2]->shading[1] = newShading;
        
        uptritable->miny = midXYZ[1];
        uptritable->maxy = maxXYZ[1];
        uptritable->midy = midXYZ[1];
        ColorTri(uptritable, buffer, screen);
        
        downtritable->edges[0] = AnalyzePoints(midXYZ[0], midXYZ[1], minXYZ[0], minXYZ[1]); // left bound
        downtritable->edges[0]->Z[0] = midXYZ[2];
        downtritable->edges[0]->Z[1] = minXYZ[2];
        downtritable->edges[0]->colors[0][0] = midColor[0];
        downtritable->edges[0]->colors[0][1] = midColor[1];
        downtritable->edges[0]->colors[0][2] = midColor[2];
        downtritable->edges[0]->colors[1][0] = minColor[0];
        downtritable->edges[0]->colors[1][1] = minColor[1];
        downtritable->edges[0]->colors[1][2] = minColor[2];
        downtritable->edges[0]->shading[0] = midShading;
        downtritable->edges[0]->shading[1] = minShading;
        
        downtritable->edges[1] = AnalyzePoints(newXYZ[0], newXYZ[1], minXYZ[0], minXYZ[1]); // right bound
        downtritable->edges[1]->Z[0] = newXYZ[2];
        downtritable->edges[1]->Z[1] = minXYZ[2];
        downtritable->edges[1]->colors[0][0] = newColor[0];
        downtritable->edges[1]->colors[0][1] = newColor[1];
        downtritable->edges[1]->colors[0][2] = newColor[2];
        downtritable->edges[1]->colors[1][0] = minColor[0];
        downtritable->edges[1]->colors[1][1] = minColor[1];
        downtritable->edges[1]->colors[1][2] = minColor[2];
        downtritable->edges[1]->shading[0] = newShading;
        downtritable->edges[1]->shading[1] = minShading;
        
        downtritable->edges[2] = AnalyzePoints(midXYZ[0], midXYZ[1], newXYZ[0], newXYZ[1]);
        downtritable->edges[2]->Z[0] = midXYZ[2];
        downtritable->edges[2]->Z[1] = newXYZ[2];
        downtritable->edges[2]->colors[0][0] = midColor[0];
        downtritable->edges[2]->colors[0][1] = midColor[1];
        downtritable->edges[2]->colors[0][2] = midColor[2];
        downtritable->edges[2]->colors[1][0] = newColor[0];
        downtritable->edges[2]->colors[1][1] = newColor[1];
        downtritable->edges[2]->colors[1][2] = newColor[2];
        downtritable->edges[2]->shading[0] = midShading;
        downtritable->edges[2]->shading[1] = newShading;
        
        downtritable->miny = minXYZ[1];
        downtritable->maxy = newXYZ[1];
        downtritable->midy = newXYZ[1];
        ColorTri(downtritable, buffer, screen);
    }
    
    delete downtritable;
    delete uptritable;
    delete intersectedge;
}

void CalculateShading(Camera c, Triangle* t)
{
    //do shading store it in the shading variable for triangles
    //use the global lighting paramaters thingy
    double LdotN;
    double v[3];
    double r[3];
    double RdotV;
    double specular;
    double diffuse;
    double ambient;
    
    for(int i=0; i<3; i++)
    {
        LdotN = dotProduct(lp.lightDir, t->normals[i]);
        
        v[0] = c.position[0] - t->X[i];
        v[1] = c.position[1] - t->Y[i];
        v[2] = c.position[2] - t->Z[i];
        
        r[0] = (2*(LdotN)*(t->normals[i][0])) - lp.lightDir[0];
        r[1] = (2*(LdotN)*(t->normals[i][1])) - lp.lightDir[1];
        r[2] = (2*(LdotN)*(t->normals[i][2])) - lp.lightDir[2];
        
        normalize(v);
        normalize(r);
        
        RdotV = dotProduct(r, v);
        
        diffuse = lp.Kd*fabs(LdotN);
        ambient = lp.Ka;
        specular = lp.Ks * pow(RdotV, lp.alpha);
        
        if((isnan(specular)) || (specular < 0))
        {
            specular = 0;
        }
        
        
        t->shading[i] = ( (ambient) + (diffuse) + (specular) );
        
        //t->shading[i] = ambient;
        //t->shading[i] = diffuse;
        //t->shading[i] = specular;
    }
    
    
}

void AnalyzeTrianglesAndColor(Triangle* tri, unsigned char* buffer, Screen screen, Matrix trans, Camera c)
{

    //printf("\n");
    Triangle* t = new Triangle();
    t = tri;
    EdgeTable* triedges = new EdgeTable();

    //DO SHADING ON TRI HERE PROLLY
    CalculateShading(c, t);
    
    //printf("Shading for vertex 1: %f\n", t->shading[0]);
    //printf("Shading for vertex 2: %f\n", t->shading[1]);
    //printf("Shading for vertex 3: %f\n", t->shading[2]);
    
    double temppoints[4];
    double newpoints[4];
    
    for(int i =0; i<3; i++)
    {
        temppoints[0] = t->X[i];
        temppoints[1] = t->Y[i];
        temppoints[2] = t->Z[i];
        temppoints[3] = 1;
        
        trans.TransformPoint(temppoints, newpoints);
        divVector(newpoints, newpoints[3]);
        
        t->X[i] = newpoints[0];
        t->Y[i] = newpoints[1];
        t->Z[i] = newpoints[2];
    }
    
    //printf("Vertex 1, X: %f, Y: %f\n", t->X[0], t->Y[0]);
    //printf("Vertex 2, X: %f, Y: %f\n", t->X[1], t->Y[1]);
    //printf("Vertex 3, X: %f, Y: %f\n", t->X[2], t->Y[2]);
    
    
    //Generate Edge 1
    triedges->edges[0] = AnalyzePoints(t->X[0], t->Y[0], t->X[1], t->Y[1]);
    triedges->edges[0]->Z[0] = t->Z[0];
    triedges->edges[0]->Z[1] = t->Z[1];
    triedges->edges[0]->colors[0][0] = t->colors[0][0];
    triedges->edges[0]->colors[0][1] = t->colors[0][1];
    triedges->edges[0]->colors[0][2] = t->colors[0][2];
    triedges->edges[0]->colors[1][0] = t->colors[1][0];
    triedges->edges[0]->colors[1][1] = t->colors[1][1];
    triedges->edges[0]->colors[1][2] = t->colors[1][2];
    triedges->edges[0]->shading[0] = t->shading[0];
    triedges->edges[0]->shading[1] = t->shading[1];

    //Generate Edge 2
    triedges->edges[1] = AnalyzePoints(t->X[0], t->Y[0], t->X[2], t->Y[2]);
    triedges->edges[1]->Z[0] = t->Z[0];
    triedges->edges[1]->Z[1] = t->Z[2];
    triedges->edges[1]->colors[0][0] = t->colors[0][0];
    triedges->edges[1]->colors[0][1] = t->colors[0][1];
    triedges->edges[1]->colors[0][2] = t->colors[0][2];
    triedges->edges[1]->colors[1][0] = t->colors[2][0];
    triedges->edges[1]->colors[1][1] = t->colors[2][1];
    triedges->edges[1]->colors[1][2] = t->colors[2][2];
    triedges->edges[1]->shading[0] = t->shading[0];
    triedges->edges[1]->shading[1] = t->shading[2];
    
    //Generate edge 3
    triedges->edges[2] = AnalyzePoints(t->X[1], t->Y[1], t->X[2], t->Y[2]);
    triedges->edges[2]->Z[0] = t->Z[1];
    triedges->edges[2]->Z[1] = t->Z[2];
    triedges->edges[2]->colors[0][0] = t->colors[1][0];
    triedges->edges[2]->colors[0][1] = t->colors[1][1];
    triedges->edges[2]->colors[0][2] = t->colors[1][2];
    triedges->edges[2]->colors[1][0] = t->colors[2][0];
    triedges->edges[2]->colors[1][1] = t->colors[2][1];
    triedges->edges[2]->colors[1][2] = t->colors[2][2];
    triedges->edges[2]->shading[0] = t->shading[1];
    triedges->edges[2]->shading[1] = t->shading[2];
    
    //Find min and max
    triedges->maxy = *std::max_element(t->Y, t->Y+3);
    triedges->miny = *std::min_element(t->Y, t->Y+3);
    
    // 3 cases to find median as it can't be max or min for an arbitary triangle
    if((t->Y[0] != triedges->maxy) && (t->Y[0] != triedges->miny))
    {
        triedges->midy = t->Y[0];
    }
    else if((t->Y[1] != triedges->maxy) && (t->Y[1] != triedges->miny))
    {
        triedges->midy = t->Y[1];
    }
    else
    {
        triedges->midy = t->Y[2];
    }
    
    AnalyzeEdges(triedges, buffer, screen);
    
    delete triedges;
}

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1e_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    cerr << "Done reading" << endl;
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();
    
    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkDoubleArray *var = (vtkDoubleArray *) pd->GetPointData()->GetArray("hardyglobal");
    double *color_ptr = var->GetPointer(0);
    //vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    //float *color_ptr = var->GetPointer(0);
    vtkFloatArray *n = (vtkFloatArray *) pd->GetPointData()->GetNormals();
    float *normals = n->GetPointer(0);
    std::vector<Triangle> tris(numTris);
    vtkIdType npts;
    vtkIdType *ptIds;
    int idx;
    for (idx = 0, cells->InitTraversal() ; cells->GetNextCell(npts, ptIds) ; idx++)
    {
        if (npts != 3)
        {
            cerr << "Non-triangles!! ???" << endl;
            exit(EXIT_FAILURE);
        }
        double *pt = NULL;
        pt = pts->GetPoint(ptIds[0]);
        tris[idx].X[0] = pt[0];
        tris[idx].Y[0] = pt[1];
        tris[idx].Z[0] = pt[2];
#ifdef NORMALS
        tris[idx].normals[0][0] = normals[3*ptIds[0]+0];
        tris[idx].normals[0][1] = normals[3*ptIds[0]+1];
        tris[idx].normals[0][2] = normals[3*ptIds[0]+2];
#endif
        pt = pts->GetPoint(ptIds[1]);
        tris[idx].X[1] = pt[0];
        tris[idx].Y[1] = pt[1];
        tris[idx].Z[1] = pt[2];
#ifdef NORMALS
        tris[idx].normals[1][0] = normals[3*ptIds[1]+0];
        tris[idx].normals[1][1] = normals[3*ptIds[1]+1];
        tris[idx].normals[1][2] = normals[3*ptIds[1]+2];
#endif
        pt = pts->GetPoint(ptIds[2]);
        tris[idx].X[2] = pt[0];
        tris[idx].Y[2] = pt[1];
        tris[idx].Z[2] = pt[2];
#ifdef NORMALS
        tris[idx].normals[2][0] = normals[3*ptIds[2]+0];
        tris[idx].normals[2][1] = normals[3*ptIds[2]+1];
        tris[idx].normals[2][2] = normals[3*ptIds[2]+2];
#endif
        
        // 1->2 interpolate between light blue, dark blue
        // 2->2.5 interpolate between dark blue, cyan
        // 2.5->3 interpolate between cyan, green
        // 3->3.5 interpolate between green, yellow
        // 3.5->4 interpolate between yellow, orange
        // 4->5 interpolate between orange, brick
        // 5->6 interpolate between brick, salmon
        double mins[7] = { 1, 2, 2.5, 3, 3.5, 4, 5 };
        double maxs[7] = { 2, 2.5, 3, 3.5, 4, 5, 6 };
        unsigned char RGB[8][3] = { { 71, 71, 219 },
            { 0, 0, 91 },
            { 0, 255, 255 },
            { 0, 128, 0 },
            { 255, 255, 0 },
            { 255, 96, 0 },
            { 107, 0, 0 },
            { 224, 76, 76 }
        };
        for (int j = 0 ; j < 3 ; j++)
        {
            float val = color_ptr[ptIds[j]];
            int r;
            for (r = 0 ; r < 7 ; r++)
            {
                if (mins[r] <= val && val < maxs[r])
                break;
            }
            if (r == 7)
            {
                cerr << "Could not interpolate color for " << val << endl;
                exit(EXIT_FAILURE);
            }
            double proportion = (val-mins[r]) / (maxs[r]-mins[r]);
            tris[idx].colors[j][0] = (RGB[r][0]+proportion*(RGB[r+1][0]-RGB[r][0]))/255.0;
            tris[idx].colors[j][1] = (RGB[r][1]+proportion*(RGB[r+1][1]-RGB[r][1]))/255.0;
            tris[idx].colors[j][2] = (RGB[r][2]+proportion*(RGB[r+1][2]-RGB[r][2]))/255.0;
        }
    }
    
    return tris;
}


int main()
{
    vtkImageData *image = NewImage(1000, 1000);
    unsigned char *buffer = (unsigned char *) image->GetScalarPointer(0,0,0);
    int npixels = 1000*1000;
    double zbuffer[npixels];
    char filename[128];
    std::vector<Triangle> triangles = GetTriangles();
    std::vector<Triangle> copytris = triangles;
    int ntris = triangles.size();
    Camera camera;
    Screen screen;
    screen.zbuffer = zbuffer;
    screen.buffer = buffer;
    screen.width = 1000;
    screen.height = 1000;
    Matrix vt;
    Matrix dt;
    Matrix ct;
    Matrix trans;
    
    for(int c = 0; c<1; c++)
    {
        for (int i = 0 ; i < npixels*3 ; i++)
            buffer[i] = 0;
        for (int i = 0 ; i < npixels ; i++)
            zbuffer[i] = -1;
        
        camera = GetCamera(c, 1000);
        ct = camera.CameraTransform();
        vt = camera.ViewTransform();
        dt = camera.DeviceTransform();
        
        std::copy(&triangles[0], &triangles[0]+ntris, &copytris[0]);
        trans = Matrix::ComposeMatrices(Matrix::ComposeMatrices(ct, vt), dt);
    
        for(int i = 0; i<ntris; i++)
        {
            AnalyzeTrianglesAndColor(&copytris[i], buffer, screen, trans, camera);
        }
        
        
        
        if(c==0)
        {
            WriteImage(image, "frame000");
        }
        /*
        if(c<10)
        {
            sprintf(filename, "images/frame00%d", c);
        }
        else if(c<100 && c>=10)
        {
            sprintf(filename, "images/frame0%d", c);
        }
        else if(c>=100)
        {
            sprintf(filename, "images/frame%d", c);
        }
        WriteImage(image, filename);
        */
        
    }
}
